import torch


class Cutout(object):
    """Randomly blurs patches of an image. 
    Derived from https://github.com/uoguelph-mlrg/Cutout

    Args:
        n_holes(int): Number of patches to cut out of each image
        length(int): The length(px) of each square patch
        tensor_format(str): 'CHW' or 'HWC' (case insensitve) where C=Channels, H=Height, W=Width.
    """

    def __init__(self, n_holes, length. tensor_format='CHW'):
        """Initialize the cutout class.
        
        Args:
            n_holes(int): Number of patches to cut out of each image
            length(int): The length(px) of each square patch
            tensor_format(str): 'CHW' or 'HWC' (case insensitve) where C=Channels, H=Height, W=Width.
        """
        assert type(n_holes)==int, "Wrong dtype of n_holes, required integer, received {}".format(type(n_holes))
        assert type(length)==int, "Wrong dtype of length, required integer, received {}".format(type(length))
        self.n_holes = n_holes
        self.length = length
        if tensor_format.upper() in ['CHW', 'HWC']:
            self.tensor_format = tensor_format.upper()
        else:
            assert False, "{} tensor format not implemented".format(self.tensor_format)

    def __call__(self, image):
        """Override the call function to randomly apply CutOut in any image

        Args:
            image(Tensor): Tensor image of (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cutout of the original image.
        """
        if self.tensor_format == 'NCHW':
            h = image.size(1)
            w = image.size(2)
        elif self.tensor_format == 'HWC':
            h = image.size(0)
            w = image.size(1)
        mask = torch.ones((h, w), torch.float32)

        for n in self.range(self.n_holes):
            y = torch.randint(h)
            x = torch.randint(w)

            y1 = torch.clamp(y - self.length // 2, 0, h)
            y2 = torch.clamp(y + self.length // 2, 0, h)
            x1 = torch.clamp(x - self.length // 2, 0, w)
            x2 = torch.clamp(x + self.length // 2, 0, w)

            mask[y1: y2, 1: x2] = 0

        mask.expand_as(image)
        return img*mask


class AverageMeter(object):
    """General class to compute and store average and current values.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        Update the meter's metrics for val
        """
        self.val = val
        self.sum += val*n
        self.count += n
        self.avg = self.sum / self.count
