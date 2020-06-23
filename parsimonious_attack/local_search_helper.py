import heapq
import math

import torch
from torch import nn
from torch.nn import functional as F

from config import IN_CHANNELS


def gather_nd(params, indices):
    '''
    the input indices must be a 2d tensor in the form of [[a,b,..,c],...],
    which represents the location of the elements.
    '''
    # Normalize indices values
    params_size = list(params.size())

    assert len(indices.size()) == 2
    assert len(params_size) >= indices.size(1)

    # Generate indices
    indices = indices.t().long()
    ndim = indices.size(0)
    idx = torch.zeros_like(indices[0]).long()
    m = 1

    for i in range(ndim)[::-1]:
        idx += indices[i] * m
        m *= params.size(i)

    params = params.reshape((-1, *tuple(torch.tensor(params.size()[ndim:]))))
    return params[idx]

class LocalSearchHelper(object):
    """A helper for local seearch algorithm.
      Note that since heapq library only supports min heap, we flip the sign of loss function.
    """
    def __init__(self, model, args):
        """Initialize local search helper.

        Args:
          model: TensorFlow model
          loss_func: str, the type of loss function
          epsilon: int, the maximum perturbation of pixel value
        """
        # Hyperparameter Setting
        self.epsilon = args.epsilon
        self.max_iters = args.max_iters
        self.targeted = args.targeted
        self.loss_func = args.loss_func
        # Network Setting
        self.model = model
        self.softmax = nn.Softmax(dim=1)
        self.in_channels= IN_CHANNELS[args.dataset]

    def get_logits(self, image):
        return self.model(image)

    def get_loss(self, logits, labels):
        probs = self.softmax(logits)
        batch_num = torch.arange(0, probs.size(0)).cuda()
        indices = torch.stack([batch_num, labels], dim=1)
        ground_truth_probs = gather_nd(params=probs, indices=indices)
        top_2 = torch.topk(probs,2,dim=1)
        max_indices = torch.where(top_2.indices[:, 0].eq(labels), top_2.indices[:, 1], top_2.indices[:, 0])
        max_indices = torch.stack([batch_num, max_indices], dim=1)
        max_probs = gather_nd(params=probs, indices=max_indices)
        if self.targeted:
            if self.loss_func == "xent":
                loss_val = F.cross_entropy(logits, labels)
            elif self.loss_func == "cw":
                loss_val = torch.log(max_probs + 1e-10) - torch.log(ground_truth_probs+1e-10)
        else:
            if self.loss_func == "xent":
                loss_val = - F.cross_entropy(logits, labels)
            elif self.loss_func == "cw":
                loss_val = torch.log(ground_truth_probs+1e-10) - torch.log(max_probs + 1e-10)
        return loss_val

    def _perturb_image(self, image, noise):
        adv_image = image + noise
        adv_image = torch.clamp(adv_image, 0, 1)
        return adv_image

    def _flip_noise(self, noise, block):
        """Filp the sign of perturbation on a block.
            Args:
              noise: numpy array of size [1, 3, 32, 32], a noise
              block: [upper_left, lower_right, channel], a block

            Returns:
              noise_new: numpy array with size [1,3, 32, 32], an updated noise
        """
        noise_new = noise.clone()
        upper_left, lower_right, channel = block
        noise_new[0, channel, upper_left[0]:lower_right[0], upper_left[1]:lower_right[1]] *= -1
        return noise_new

    def perturb(self, image, noise, label, blocks):
        """Update a noise with local search algorithm.

        Args:
          image: torch array of size [1, 3, 32, 32], an original image
          noise: torch array of size [1, 3, 32, 32], a noise
          label: torch array of size [1], the label of the image (or target label)
          blocks: list, a set of blocks

        Returns:
          noise: numpy array of size [1, 256, 256, 3], an updated noise
          num_queries: int, the number of queries
          curr_loss: float, the value of loss function
          success: bool, True if attack is successful
        """
        # Local variables
        priority_queue = []
        num_queries = 0
        img_size = image.size(2)
        # Check if a block is in the working set or not
        A = torch.zeros((len(blocks)), dtype=torch.int32)
        for i, block in enumerate(blocks):
            upper_left, _, channel = block
            x = upper_left[0]
            y = upper_left[1]
            # If the sign of perturbation on the block is positive,
            # which means the block is in the working set, then set A to 1
            if noise[0, channel, x, y] > 0:
                A[i] = 1
        # Calculate the current loss
        image_batch = self._perturb_image(image, noise)
        label_batch = label.clone()
        logits = self.get_logits(image_batch)
        preds = torch.argmax(logits, 1)
        losses = self.get_loss(logits, label_batch)
        num_queries += 1
        curr_loss = losses[0]
        # Early stopping
        if self.targeted:
            if torch.all(preds.eq(label)).item():
                return noise, num_queries, curr_loss, True
        else:
            if not torch.all(preds.eq(label)).item():
                return noise, num_queries, curr_loss, True
        # Main loop
        for _ in range(self.max_iters):
            # Lazy greedy insert、
            indices = torch.nonzero(A==0).view(-1)
            batch_size =  100
            num_batches = int(math.ceil(indices.size(0) / batch_size))
            for ibatch in range(num_batches):
                bstart = ibatch * batch_size
                bend = min(bstart + batch_size, indices.size(0))
                image_batch = torch.zeros([bend - bstart,self.in_channels,img_size,img_size]).float().cuda()
                noise_batch = torch.zeros([bend - bstart,self.in_channels,img_size,img_size]).float().cuda()
                label_batch = label.repeat(bend - bstart)
                for i, idx in enumerate(indices[bstart:bend]):
                    idx = idx.item()
                    noise_batch[i:i + 1, ...] = self._flip_noise(noise, blocks[idx])
                    image_batch[i:i + 1, ...] = self._perturb_image(image, noise_batch[i:i + 1, ...])
                logits = self.get_logits(image_batch)
                preds = torch.argmax(logits, 1)
                losses = self.get_loss(logits, label_batch)
                # Early stopping
                success_indices = torch.nonzero((preds == label_batch).long()).view(-1) if self.targeted else torch.nonzero((preds!=label_batch).long()).view(-1)
                if success_indices.size(0) > 0:
                    noise[0, ...] = noise_batch[success_indices[0], ...]
                    curr_loss = losses[success_indices[0]]
                    num_queries += success_indices[0].item() + 1
                    return noise, num_queries, curr_loss, True

                num_queries += bend - bstart
                # Push into the priority queue
                for i in range(bend - bstart):
                    idx = indices[bstart + i]
                    margin = losses[i] - curr_loss
                    heapq.heappush(priority_queue, (margin.item(), idx))
            # Pick the best element and insert it into the working set
            if len(priority_queue) > 0:
                best_margin, best_idx = heapq.heappop(priority_queue)
                curr_loss += best_margin
                noise = self._flip_noise(noise, blocks[best_idx])
                A[best_idx] = 1
            # Add elements into the working set
            while len(priority_queue) > 0:
                # Pick the best element
                cand_margin, cand_idx = heapq.heappop(priority_queue)
                # Re-evalulate the element
                image_batch = self._perturb_image(
                    image, self._flip_noise(noise, blocks[cand_idx]))
                label_batch = label.clone()
                logits = self.get_logits(image_batch)
                preds = torch.argmax(logits, 1)
                losses = self.get_loss(logits, label_batch)
                num_queries += 1
                margin = losses[0] - curr_loss
                # If the cardinality has not changed, add the element
                if len(priority_queue) == 0 or margin.item() <= priority_queue[0][0]:
                    # If there is no element that has negative margin, then break
                    if margin.item() > 0:
                        break
                    # Update the noise
                    curr_loss = losses[0]
                    noise = self._flip_noise(noise, blocks[cand_idx])
                    A[cand_idx] = 1
                    # Early stopping
                    if self.targeted:
                        if torch.all(preds.eq(label)).item():
                            return noise, num_queries, curr_loss, True
                    else:
                        if not torch.all(preds.eq(label)).item():
                            return noise, num_queries, curr_loss, True
                # If the cardinality has changed, push the element into the priority queue
                else:
                    heapq.heappush(priority_queue, (margin.item(), cand_idx))
            priority_queue = []
            # Lazy greedy delete
            indices  = torch.nonzero(A == 1).view(-1)
            batch_size = 100
            num_batches = int(math.ceil(indices.size(0) / batch_size))
            for ibatch in range(num_batches):
                bstart = ibatch * batch_size
                bend = min(bstart + batch_size, indices.size(0))

                image_batch = torch.zeros([bend - bstart,self.in_channels, img_size, img_size]).float().cuda()
                noise_batch = torch.zeros([bend - bstart,self.in_channels, img_size, img_size]).float().cuda()
                label_batch = label.repeat(bend - bstart)
                for i, idx in enumerate(indices[bstart:bend]):
                    noise_batch[i:i + 1, ...] = self._flip_noise(noise, blocks[idx])
                    image_batch[i:i + 1, ...] = self._perturb_image(image, noise_batch[i:i+1, ...])
                logits = self.get_logits(image_batch)
                preds = torch.argmax(logits, 1)
                losses = self.get_loss(logits, label_batch)
                success_indices = torch.nonzero((preds == label_batch).long()).view(-1) if self.targeted else torch.nonzero((preds!=label_batch).long()).view(-1)
                if success_indices.size(0) > 0:
                    noise[0, ...] = noise_batch[success_indices[0], ...]
                    curr_loss = losses[success_indices[0]]
                    num_queries += success_indices[0].item() + 1
                    return noise, num_queries, curr_loss, True
                num_queries += bend - bstart
                # Push into the priority queue
                for i in range(bend - bstart):
                    idx = indices[bstart + i].item()
                    margin = losses[i] - curr_loss
                    heapq.heappush(priority_queue, (margin.item(), idx))

            # Pick the best element and remove it from the working set
            if len(priority_queue) > 0:
                best_margin, best_idx = heapq.heappop(priority_queue)
                curr_loss += best_margin
                noise = self._flip_noise(noise, blocks[best_idx])
                A[best_idx] = 0
            # Delete elements from the working set
            while len(priority_queue) > 0:
                # pick the best element
                cand_margin, cand_idx = heapq.heappop(priority_queue)
                # Re-evalulate the element
                image_batch = self._perturb_image(image, self._flip_noise(noise, blocks[cand_idx]))
                label_batch = label.clone()
                logits = self.get_logits(image_batch)
                preds = torch.argmax(logits, 1)
                losses = self.get_loss(logits, label_batch)
                num_queries += 1
                margin = losses[0] - curr_loss
                # If the cardinality has not changed, remove the element
                if len(priority_queue) == 0 or margin.item() <= priority_queue[0][0]:
                    # If there is no element that has negative margin, then break
                    if margin.item() > 0:
                        break
                    # Update the noise
                    curr_loss = losses[0]
                    noise = self._flip_noise(noise, blocks[cand_idx])
                    A[cand_idx] = 0
                    # Early stopping
                    if self.targeted:
                        if torch.all(preds.eq(label)).item():
                            return noise, num_queries, curr_loss, True
                    else:
                        if not torch.all(preds.eq(label)).item():
                            return noise, num_queries, curr_loss, True
                else:
                    heapq.heappush(priority_queue, (margin.item(), cand_idx))

            priority_queue = []
        return noise, num_queries, curr_loss, False