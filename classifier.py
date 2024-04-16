from torch import nn
class Classifier(nn.Module):
    def __init__(self, input_size=512, hidden_dim1=256, hidden_dim2=128):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_size, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.classifier(x)
        return x