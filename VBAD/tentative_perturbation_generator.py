import torch
from torch import nn

class TentativePerturbationGenerator():
    def __init__(self, extractors, norm, part_size=100, preprocess=True, device=0):
        self.r = None
        self.extractors = extractors
        self.norm = norm
        self.part_size = part_size  # The number of frames are processed each time
        self.preprocess = preprocess
        self.device = device

    def set_targeted_params(self, target_vid, random_mask=1.):
        self.target = True
        self.random_mask = random_mask
        self.target_feature = []
        with torch.no_grad():
            target_vid = target_vid.clone().cuda(self.device)
            if self.preprocess:
                mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=target_vid.get_device())[None, :,
                       None, None]
                std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=target_vid.get_device())[None, :,
                      None, None]

                target_vid = target_vid.sub_(mean).div_(std)

            for extractor in self.extractors:
                outputs = extractor(target_vid)
                self.target_feature.append(outputs[0].view((outputs[0].size(0), -1)))

    def set_untargeted_params(self, ori_video, random_mask=1., translate=0., scale=1.):
        self.target = False
        self.translate = translate
        self.scale = scale
        self.random_mask = random_mask
        self.target_feature = []
        with torch.no_grad():
            ori_video = ori_video.clone().cuda(self.device)
            for extractor in self.extractors:
                outputs = extractor(ori_video)
                output_size = outputs[0].size()
                del outputs
                r = torch.randn(output_size, device=self.device) * self.scale + self.translate
                r = torch.where(r >= 0, r, -r)
                self.target_feature.append(r.view((output_size[0], -1)))

    # vid shape: [num_frames, c, w, h]
    def create_adv_directions(self, vid, random=True):
        vid = vid.clone().cuda(self.device)
        assert hasattr(self, 'target'), 'Error, AdvDirectionCreator\' mode unset'
        start_idx = 0
        adv_directions = []
        part_size = self.part_size
        while start_idx < vid.size(0):
            adv_directions.append(self.backpropagate2frames(vid[start_idx:min(start_idx + part_size, vid.size(0))],
                                                            start_idx, start_idx + part_size, random))
            start_idx += part_size
        adv_directions = torch.cat(adv_directions, 0)
        return adv_directions

    def normalize(self, t):
        assert len(t.shape) == 4
        norm_vec = torch.sqrt(t.pow(2).sum(dim=[1, 2, 3])).view(-1, 1, 1, 1)
        norm_vec += (norm_vec == 0).float() * 1e-8
        return norm_vec

    def backpropagate2frames(self, part_vid, start_idx, end_idx, random):
        part_vid.requires_grad = True
        processed_vid = part_vid.clone()
        if self.preprocess:
            mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=part_vid.get_device())[None, :, None,
                   None]
            std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=part_vid.get_device())[None, :, None,
                  None]
            processed_vid = processed_vid.sub_(mean).div_(std)

        for idx, extractor in enumerate(self.extractors):
            perturb_loss = 0
            o = extractor(processed_vid)[0]
            o = o.view((o.size(0), -1))
            if self.target:
                if random:
                    mask = torch.rand_like(o) <= self.random_mask
                    perturb_loss += nn.MSELoss(reduction='mean')(torch.masked_select(o, mask),
                                                                             torch.masked_select(
                                                                                 self.target_feature[idx][
                                                                                 start_idx:end_idx], mask))
                else:
                    perturb_loss += nn.MSELoss(reduction='mean')(o, self.target_feature[idx][
                                                                                start_idx:end_idx])
            else:
                r = torch.randn_like(o) * self.scale + self.translate
                r = torch.where(r >= 0, r, -r)
                perturb_loss += nn.MSELoss(reduction='mean')(o, r)

            perturb_loss.backward()
            extractor.zero_grad()
        if self.norm == "linf":
            grad = torch.sign(part_vid.grad)
        elif self.norm == "l2":
            grad = part_vid.grad.clone()
        return grad

    def __call__(self, vid):
        return self.create_adv_directions(vid)
