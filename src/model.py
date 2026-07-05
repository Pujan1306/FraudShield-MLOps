import torch
import torch.nn as nn

class FraudClassificationModel(nn.Module):
    def __init__(self, input_dim: int = 30):
        super(FraudClassificationModel, self).__init__()
        self.network = nn.Sequential(
            #Block 1: Input to Hidden
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            #Block 2: Hidden to Hidden
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            #Block 3: Hidden to Hidden
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),

            #Block 4: Output Layer
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.network(x)
        return x

if __name__ == "__main__":
    mock_batch = torch.rand(5, 30)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FraudClassificationModel(input_dim=30).to(device)
    output = model(mock_batch)
    print(f"✅ nn.Sequential Model check passed. Output shape: {output.shape} (Expected: [5, 1]) and Devide: {device}")