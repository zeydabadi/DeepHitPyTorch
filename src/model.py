import torch.nn as nn

class BasicDeepHitMLP(nn.Module):
    def __init__(self, in_features, num_events, num_time_steps,
                 hidden_units=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, num_events * num_time_steps)
        )
        self.num_events = num_events
        self.num_time_steps = num_time_steps

    def forward(self, x):
        out = self.net(x)
        return out.view(x.size(0), self.num_events, self.num_time_steps)
