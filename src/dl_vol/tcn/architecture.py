import torch.nn as nn
import torch.nn.functional as F


class TemporalBlock(nn.Module):
    """Minimal causal temporal block.

    Left-padding, convolution, ReLU, dropout. No double blocks like Bai. et al. 2018.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolutional kernel.
        dilation (int): Dilation factor for the convolution.
        dropout (float): Dropout rate.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation,
        dropout=0.2,
    ):
        super().__init__()

        self.left_padding = (kernel_size - 1) * dilation

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=0, dilation=dilation)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        )

        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity="relu")

        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        residual = self.downsample(x) if self.downsample is not None else x

        # Left-pad for causality
        out = F.pad(x, (self.left_padding, 0))
        out = self.conv(out)
        out = self.relu(out)
        out = self.dropout(out)

        return residual + out


class TCN(nn.Module):
    """Temporal Convolutional Network (TCN) for time series forecasting.

    Args:
        num_inputs (int): Number of input channels.
        num_channels (list of int): List of output channels for each temporal block.
        kernel_size (int): Size of the convolutional kernel.
        dropout (float): Dropout rate.
    """

    def __init__(
        self,
        num_inputs,
        num_channels,
        kernel_size=3,
        dropout=0.2,
    ):
        super().__init__()

        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation_size = 2**i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]

            layers.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    dilation=dilation_size,
                    dropout=dropout,
                )
            )

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCNForecast(nn.Module):
    """TCN-based model for time series forecasting.

    Args:
        tcn (TCN): An instance of the TCN class.
        output_size (int): Number of output channels for the forecast.
    """

    def __init__(self, tcn, output_size):
        super().__init__()
        self.tcn = tcn
        self.linear = nn.Linear(tcn.network[-1].conv.out_channels, output_size)

    def forward(self, x):
        tcn_out = self.tcn(x)
        # Take the last time step's output for forecasting
        last_time_step = tcn_out[:, :, -1]
        forecast = self.linear(last_time_step)
        return forecast
