import torch
from .utils import read_model, extract_layers, get_layer_properties

class ArithmeticIntensity(object):
    """
    Calculates the arithmetic intensity of a given PyTorch models.
    """
    def __init__(self, model=None, path=None, input_dims=None, data_format="NCHW"):
        """
        """
        assert model is not None or path is not None, "Requires either model or path"
        assert input_dims is not None, "Enter valid input input dims, received empty ()"
        assert len(input_dims) == 3 or len(input_dims) == 4, "Invalid input dimensions"
        assert data_format in ["NHWC", "NCHW", "HWC", "CHW"], 'Invalid data format, should be in ["NHWC", "NCHW", "HWC", "CHW"]'
        if model:
            self.model = model
        else:
            try:
                self.model = read_model(path)
            except:
                assert False, "Read model error"
        self.layers = extract_layers(model, [])
        self.data_format = data_format
        if len(data_format) == 3:
            data_format = 'N' + data_format
        if len(input_dims) == 4:
            self.batch_size = input_dims[0]
            if data_format=="NCHW":
                self.input_dims = input_dims
            elif data_format=="NHWC":
                self.input_dims = (self.batch_size, input_dims[3], *input_dims[1:3])
        else:
            self.batch_size = 1
            if data_format == "NHWC":
                self.input_dims = (self.batch_size, input_dims[3], *input_dims[1:3])
            else:
                self.input_dims = (self.batch_size, *input_dims)

    def get_metrics(self):
        """

        """
        dummy = torch.ones(1, *self.input_dims[1:])
        print("Dummy: ", dummy.size())
        ai_sum = 0
        macs_sum = 0
        for layer in self.layers:
            if layer.__str__().find('Conv2d') > -1:
                print(layer)
                ai, macs = get_layer_properties(layer, self.batch_size, dummy.size()[2:])
                ai_sum += ai
                macs_sum += macs
            try:
                dummy = layer(dummy)
            except:
                break
        print("Arithmetic Intensity: ", ai_sum)
        print("MACS: ", macs_sum)
        return ai_sum, macs_sum