import torch
import sys


class Cutout(object):
    """Randomly blurs patches of an image. 
    Derived from https://github.com/uoguelph-mlrg/Cutout

    Args:
        n_holes(int): Number of patches to cut out of each image
        length(int): The length(px) of each square patch
        tensor_format(str): 'CHW' or 'HWC' (case insensitve) where C=Channels, H=Height, W=Width.
    """

    def __init__(self, n_holes, length, p=0.5, tensor_format='CHW'):
        """Initialize the cutout class.
        
        Args:
            n_holes(int): Number of patches to cut out of each image
            length(int): The length(px) of each square patch
            tensor_format(str): 'CHW' or 'HWC' (case insensitve) where C=Channels, H=Height, W=Width.
        """
        assert type(n_holes)==int, "Wrong dtype of n_holes, required integer, received {}".format(type(n_holes))
        assert type(length)==int, "Wrong dtype of length, required integer, received {}".format(type(length))
        assert p > 0 and p < 1, "Range of p in (0, 1), received {}".format(p)
        self.n_holes = n_holes
        self.length = length
        self.p = p
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
        if torch.randn(1) < self.p:
            if self.tensor_format == 'CHW':
                h = image.size(1)
                w = image.size(2)
            elif self.tensor_format == 'HWC':
                h = image.size(0)
                w = image.size(1)
            mask = torch.ones((h, w), dtype=torch.float32)

            for n in range(self.n_holes):
                y = torch.randint(h, (1, 1))
                x = torch.randint(w, (1, 1))

                y1 = torch.clamp(y - self.length // 2, 0, h)
                y2 = torch.clamp(y + self.length // 2, 0, h)
                x1 = torch.clamp(x - self.length // 2, 0, w)
                x2 = torch.clamp(x + self.length // 2, 0, w)

                mask[y1: y2, x1: x2] = 0

            mask.expand_as(image)
            image = image * mask

        return image


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


class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass