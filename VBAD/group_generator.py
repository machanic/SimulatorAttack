# This class is for patch-level operations.
class EquallySplitGrouping(object):
    def __init__(self, divide_number):
        self.length = 0
        self.dim = None
        self.divide_number = divide_number

    def initialize(self, x):
        assert x.size(-1) % self.divide_number == 0, 'frame size: {} not divided evenly by {}'.format(x.size(-1),
                                                                                                      self.divide_number)
        self.length = self.divide_number * self.divide_number * x.size(0)
        self.dim = x.size()

    def __len__(self):
        return self.length

    def apply_group_change(self, x, y):
        """
        :param x: shape = (frame_number, C, H, W)
        :param y: shape = (sub_num, frame_number * patch number * patch_number), contents are [1st frame's 1st patch, ..., 1st frame's last patch, 2nd frame's 1st patch, ..., 2nd frame's last patch]
        :return: perturbed noise vector
        """
        assert (x.size() == self.dim) and (
                (len(y.size()) == 1) or (len(y.size()) == 2)), 'x size: {}    y size:{}'.format(x.size(), y.size())
        batch_mode = False if len(y.size()) == 1 else True
        patch_size = x.size(-1) // self.divide_number
        frames_number = x.size(0)
        x_t = x.repeat((y.size(0),) + (1,) * len(x.size())) if batch_mode else x.clone()  # (sub_num, frame_number,C,H,W)  or (frame_number, C,H,W)

        for i in range(self.divide_number):
            for j in range(self.divide_number):
                patch_idx = i * self.divide_number + j
                if batch_mode:
                    patch = x_t[:, :, :, i * patch_size:(i + 1) * patch_size,
                            j * patch_size:(j + 1) * patch_size]  # shape = (sub_num, frame_number, C, patch_size, patch_size)
                    patch *= y[:, patch_idx * frames_number:(patch_idx + 1) * frames_number].view(
                        (y.size(0), frames_number) + (1,) * (len(patch.size()) - 2))  # reshape to (sub_num, frame_number, 1,1,1)
                else:
                    patch = x_t[:, :, i * patch_size:(i + 1) * patch_size,
                            j * patch_size:(j + 1) * patch_size] # frame_number,C,patch_size, patch_size
                    patch *= y[patch_idx * frames_number:(patch_idx + 1) * frames_number].view(
                        (frames_number,) + (1,) * (len(patch.size()) - 1))  # reshape to (frame_number, 1,1,1)
        return x_t
