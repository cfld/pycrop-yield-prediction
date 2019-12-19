import torch.nn as nn
from torch.autograd import Variable
import torch
import torch.nn.functional as F
from pathlib import Path

from .baseConvLSTM import ModelBase



class ConvLSTMModel(ModelBase):
    """
    A PyTorch implementation of ConvLSTM cell in framework for crop_yeild paper

    Parameters
    ----------
    in_channels: int, default=9
        Number of channels in the input data. Default taken from the number of bands in the
        MOD09A1 + the number of bands in the MYD11A2 datasets
    num_bins: int, default=32
        Number of bins in the histogram
    hidden_size: int, default=128
        The size of the hidden state. Default taken from the original repository
    rnn_dropout: float, default=0.75
        Default taken from the original paper. Note that this dropout is applied to the
        hidden state after each timestep, not after each layer (since there is only one layer)
    dense_features: list, or None, default=None.
        output feature size of the Linear layers. If None, default values will be taken from the paper.
        The length of the list defines how many linear layers are used.
    savedir: pathlib Path, default=Path('data/models')
        The directory into which the models should be saved.
    device: torch.device
        Device to run model on. By default, checks for a GPU. If none exists, uses
        the CPU
    """

    def __init__(self, in_channels=9, pix=64, hidden_size=128, rnn_dropout=0.75,
                 dense_features=None, savedir=Path('data/models'), r_loc=0.5, r_year=1.5,
                 device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):

        model = ConvLSTM(height         = pix,
                         width          = pix,
                         input_dim      = in_channels,
                         hidden_dim     = [16, 16, 16, 1],
                         kernel_size    = (3, 3),
                         num_layers     = 4,
                         batch_first    = True,
                         bias           = True,
                         return_all_layers=False)

        if dense_features is None:
            num_dense_layers = 2
        else:
            num_dense_layers = len(dense_features)
        model_weight = f'dense_layers.{num_dense_layers - 1}.weight'
        model_bias = f'dense_layers.{num_dense_layers - 1}.bias'

        super().__init__(model, model_weight, model_bias, 'rnn', savedir, r_loc, r_year, device)

    def reinitialize_model(self, time=None):
        self.model.initialize_weights()

class ConvLSTMCell(nn.Module):

    def __init__(self, height, width, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        height: int
        width: int
            Height and width of input tensor.
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """
        super(ConvLSTMCell, self).__init__()

        self.height         = height
        self.width          = width
        self.input_dim      = input_dim
        self.hidden_dim     = hidden_dim
        self.kernel_size    = kernel_size
        self.padding        = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias           = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)


    def forward(self, input, prev_state):
        h_prev, c_prev = prev_state
        combined = torch.cat((input, h_prev), dim=1) #cat on channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

        i = F.sigmoid(cc_i)
        f = F.sigmoid(cc_f)
        o = F.sigmoid(cc_o)
        g = F.sigmoid(cc_g)

        c_cur = f * c_prev * i * g
        h_cur = o * F.tanh(c_cur)

        return h_cur, c_cur

    def init_hidden(self, batch_size, device):
        state = (Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)),
                 Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)))
        state = state.to(device)

        return state

class ConvLSTM(nn.Module):

    def __init(self, height, width, input_dim, hidden_dim, kernel_size, num_layers, batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # kernel_size and hidden_dim should be lists of len(num_layers)
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim  = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('list len not same')


        self.height         = height
        self.width          = width
        self.input_dim      = input_dim
        self.kernel_size    = kernel_size
        self.num_layers     = num_layers
        self.batch_first    = batch_first
        self.bias           = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            current_input_dim = self.input_dim if i == 0 else self.hidden_dim[i-1]

            cell_list.append(ConvLSTMCell(height        = self.height,
                                          width         = self.width,
                                          input_dim     = self.input_dim,
                                          hidden_dim    = self.hidden_dim[i],
                                          kernel_size   = self.kernel_size[i],
                                          bias          = self.bias))

            self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input, hidden_state=None):
        '''

        Parameters
        ----------
        input_tensor:
            5-D Tensor of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state

        Returns
        ----------
        last_state_list, layer_output
        '''
        if not self.batch_first:
            input = input.permute(1,0,2,3,4)

        if hidden_state is None:
            hidden_state = self.get_init_states(batch_size=input.size(0))

        layer_output_list   = []
        last_state_list     = []
        seq_len             = input.size(0)
        curr_layer_input    = input

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input=cur_layer_input[:, t, :, :, :],
                                                 prev_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append((h, c))

        layer_output = layer_output_list[-1]
        if not self.batch_first:
            layer_output = layer_output.permute(1, 0, 2, 3, 4)


        return layer_output, last_state_list

    def get_init_states(self, batch_size, cuda=True):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, cuda))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size])):
            raise ValueError('kernel_size must be tuple or list of tuples')


    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param











