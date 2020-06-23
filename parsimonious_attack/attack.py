import itertools
import math

import glog as log
import torch

from parsimonious_attack.local_search_helper import LocalSearchHelper


class ParsimoniousAttack(object):
    """Parsimonious attack using local search algorithm"""

    def __init__(self, model, args):
        """Initialize attack.

        Args:
          model: TensorFlow model
          args: arguments
        """
        # Hyperparameter setting
        self.loss_func = args.loss_func
        self.max_queries = args.max_queries
        self.epsilon = args.epsilon
        self.batch_size = args.batch_size
        self.block_size = args.block_size
        self.no_hier = args.no_hier
        self.max_iters = args.max_iters
        # Create helper
        self.local_search = LocalSearchHelper(model, args)

    def _perturb_image(self, image, noise):
        """Given an image and a noise, generate a perturbed image.

        Args:
          image: numpy array of size [1, 32, 32, 3], an original image
          noise: numpy array of size [1, 32, 32, 3], a noise

        Returns:
          adv_image: numpy array of size [1, 299, 299, 3], an adversarial image
        """
        adv_image = image + noise
        adv_image = torch.clamp(adv_image, 0, 1)
        return adv_image

    def _split_block(self, upper_left, lower_right, block_size):
        """Split an image into a set of blocks.
        Note that a block consists of [upper_left, lower_right, channel]

        Args:
          upper_left: [x, y], the coordinate of the upper left of an image
          lower_right: [x, y], the coordinate of the lower right of an image
          block_size: int, the size of a block

        Return:
          blocks: list, a set of blocks
        """
        blocks = []
        xs = torch.arange(upper_left[0], lower_right[0], block_size)
        ys = torch.arange(upper_left[1], lower_right[1], block_size)
        for x, y in itertools.product(xs.detach().numpy(), ys.detach().numpy()):
            for c in range(3):
                blocks.append([[x, y], [x + block_size, y + block_size], c])
        return blocks

    def perturb(self, image, label):
        """Perturb an image.

        Args:
          image: numpy array of size [1, 3, 32, 32], an original image
          label: numpy array of size [1], the label of the image (or target label)

        Returns:
          adv_image: numpy array of size [1, 32, 32, 3], an adversarial image
          num_queries: int, the number of queries
          success: bool, True if attack is successful
        """

        adv_image = image.clone()
        num_queries = 0
        block_size = self.block_size
        upper_left = [0, 0]
        lower_right = [image.size(2), image.size(2)]
        # Split an image into a set of blocks
        blocks = self._split_block(upper_left, lower_right, block_size)
        # Initialize a noise to -epsilon
        noise = -self.epsilon * torch.ones_like(image).float().cuda()
        # Construct a batch
        num_blocks = len(blocks)
        batch_size = self.batch_size if self.batch_size > 0 else num_blocks
        curr_order = torch.randperm(num_blocks)
        # Main loop
        while True:
            num_batches = int(math.ceil(num_blocks / batch_size))
            for i in range(num_batches):
                # Pick a mini-batch
                bstart = i * batch_size
                bend = min(bstart + batch_size, num_blocks)
                blocks_batch = [blocks[curr_order[idx].item()]
                                for idx in range(bstart, bend)]
                # Run local search algorithm on the mini-batch
                noise, queries, loss, success = self.local_search.perturb(
                    image, noise, label, blocks_batch)
                num_queries += queries
                log.info("Block size: {}, batch: {}, loss: {:.4f}, num queries: {}".format(block_size, i, loss, num_queries))
                # If query count exceeds the maximum queries, then return False
                if num_queries > self.max_queries:
                    return num_queries, False
                # Generate an adversarial image
                # adv_image = self._perturb_image(image, noise)
                # If attack succeeds, return True
                if success:
                    return num_queries, True
            # If block size is >= 2, then split the image into smaller blocks and reconstruct a batch
            if not self.no_hier and block_size >= 2:
                block_size //= 2
                blocks = self._split_block(upper_left, lower_right, block_size)
                num_blocks = len(blocks)
                batch_size = self.batch_size if self.batch_size > 0 else num_blocks
                curr_order = torch.randperm(num_blocks)
            # Otherwise, shuffle the order of the batch
            else:
                curr_order = torch.randperm(num_blocks)