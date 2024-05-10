from torch import nn
class Classifier(nn.Module):
    def __init__(self, input_size=1456, hidden_dim1=728, hidden_dim2=364, hidden_dim3 = 182):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_size, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, hidden_dim3),
            nn.ReLU(),
            nn.Linear(hidden_dim3, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.classifier(x)
        return x