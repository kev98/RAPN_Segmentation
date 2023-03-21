import torch
import torch.nn as nn
import torch.nn.functional as F

# Definition of the basic cell of the convLSTM
class ConvLSTMCell(nn.Module):  # normal conv with peephole connections
    def __init__(self, input_channels, hidden_channels, kernel_size, dilation, activation_function):
        super(ConvLSTMCell, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.num_gates = 4  # f i g o
        self.padding = int((kernel_size - 1) / 2)
        self.convolution = nn.Conv2d(self.input_channels + self.hidden_channels, 4 * self.hidden_channels,
                                     self.kernel_size, stride=1, padding=self.padding, dilation=dilation)
        self.activation_function = activation_function
        self.peephole_weights = nn.Parameter(torch.zeros(3, self.hidden_channels), requires_grad=True)

    # Define the operations of an entire time step of the cell
    def forward(self, x, h, c):  # x: batch, channel, height, width; h: hidden state; c: previous cell output
        x_stack_h = torch.cat((x, h), dim=1)  # cat the input with the previous hidden state
        A = self.convolution((x_stack_h))
        split_size = int(A.shape[1] / self.num_gates)
        # inputs for: input_gate, forget_gate, output_gate, cell_output_gate
        (ai, af, ao, ag) = torch.split(A, split_size, dim=1)
        f = torch.sigmoid(af + c * self.peephole_weights[1, :, None, None])
        i = torch.sigmoid(ai + c * self.peephole_weights[0, :, None, None])
        g = self.activation_function(ag)
        # Current Cell output
        o = torch.sigmoid(ao + c * self.peephole_weights[2, :, None, None])
        new_c = f * c + i * g
        # Current Hidden State
        new_h = o * self.activation_function(new_c)
        return new_h, new_c

# Define the entire Cycle of the convLSTM on a single input of the sequence
class ConvLSTM(nn.Module):
    # input_channels corresponds to the first input feature map
    # hidden state is a list of succeeding lstm layers.
    def __init__(self, input_channels, hidden_channels, kernel_size, activation_function, device,
                 dtype, state_init, overlap, dilation=1, init='default',
                 is_stateful=True, state_img_size=None):
        super(ConvLSTM, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        if activation_function == 'prelu':
            activation_function = nn.PReLU()
        else:
            activation_function = nn.Tanh()

        self.cell = ConvLSTMCell(self.input_channels, self.hidden_channels, self.kernel_size, self.dilation,
                                      activation_function)
        self.is_stateful = is_stateful
        self.dtype = dtype
        self.device = device
        self.state_init = state_init
        self.state_img_size = state_img_size

        self.update_parameters(overlap)
        self.init_states(state_img_size, state_init) # call the function that initialize kernels

        # initialization of the convolutional weights
        if init == 'default':
            self.cell.convolution.bias.data.fill_(0)  # init all biases with 0
            nn.init.xavier_normal_(self.cell.convolution.weight.data[
                                   0 * self.cell.hidden_channels: 1 * self.cell.hidden_channels])  # sigmoid, i
            nn.init.xavier_normal_(self.cell.convolution.weight.data[
                                   1 * self.cell.hidden_channels: 2 * self.cell.hidden_channels])  # sigmoid, f
            self.cell.convolution.bias.data[1 * self.cell.hidden_channels: 2 * self.cell.hidden_channels].fill_(
                0.1)  # f bias
            nn.init.xavier_normal_(self.cell.convolution.weight.data[
                                   2 * self.cell.hidden_channels: 3 * self.cell.hidden_channels])  # sigmoid, o

        if activation_function == 'tanh':
            nn.init.xavier_normal_(self.cell.convolution.weight.data[
                                   3 * self.cell.hidden_channels: 4 * self.cell.hidden_channels])  # tanh, g
        elif activation_function in ['lrelu', 'prelu']:
            nn.init.kaiming_normal_(
                self.cell.convolution.weight.data[3 * self.cell.hidden_channels: 4 * self.cell.hidden_channels],
                nonlinearity='leaky_relu')  # lrelu, g

    def forward(self, input, states, batch_size):  # inputs shape: time_step, batch_size, channels, height, width
        new_states = states
        outputs = torch.empty(batch_size, self.hidden_channels, input.shape[2],
                              input.shape[3], dtype=self.dtype, device=self.device)
        if self.is_stateful == 0 or states is None:
            h = nn.functional.interpolate(self.h0.expand(batch_size, -1, -1, -1),
                                          size=(input.shape[2], input.shape[3]), mode='bilinear', align_corners=True)
            c = nn.functional.interpolate(self.c0.expand(batch_size, -1, -1, -1),
                                          size=(input.shape[2], input.shape[3]), mode='bilinear', align_corners=True)
          #  print("Init LSTM")
        else:
            c = states[0]
            h = states[1]

        x = input

        h, c = self.cell(x, h, c)  # to run hooks (pre, post) and .forward()

        output = h
        new_states = torch.stack((c.data, h.data))

        return output, new_states

    # initialize hidden state and cell input
    def init_states(self, state_size, state_init):

        if state_init == 'zero':
            self.h0 = nn.Parameter(torch.zeros(self.hidden_channels, state_size[0], state_size[1], dtype=self.dtype),
                                   requires_grad=False)
            self.c0 = nn.Parameter(torch.zeros(self.hidden_channels, state_size[0], state_size[1], dtype=self.dtype),
                                   requires_grad=False)
        elif state_init == 'rand':  # cell_state rand [0,1) init
            self.h0 = nn.Parameter(torch.rand(self.hidden_channels, state_size[0], state_size[1], dtype=self.dtype),
                                   requires_grad=False)
            self.c0 = nn.Parameter(torch.rand(self.hidden_channels, state_size[0], state_size[1], dtype=self.dtype),
                                   requires_grad=False)
        elif state_init == 'learn':
                self.h0 = nn.Parameter(
                    torch.zeros(self.hidden_channels, state_size[0], state_size[1], dtype=self.dtype),
                    requires_grad=True)

                self.c0 = nn.Parameter(
                    torch.zeros(self.hidden_channels, state_size[0], state_size[1], dtype=self.dtype),
                    requires_grad=True)

    def update_parameters(self, overlap):
        self.overlap = overlap


