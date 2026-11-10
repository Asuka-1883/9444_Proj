"""LSTM encoder for pedestrian motion."""

from torch import nn


class LSTMMotionEncoder(nn.Module):
    def __init__(self, hidden_size=64, layers=1, dropout=0.1):
        super().__init__()

        # Each time step has four motion features.
        self.rnn = nn.LSTM(
            input_size=4,
            hidden_size=hidden_size,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )

    def forward(self, x):
        # x: [batch, observed_steps, 4]
        sequence, (hidden, _cell) = self.rnn(x)

        # Return the format used by the shared model.
        return sequence, hidden