import torch.nn as nn

def FeedForward(input_dim: int, hidden_dim: int, output_dim: int, num_layers: int, dropout: float = 0.0):
    layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
    
    for _ in range(num_layers):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        
    layers.append(nn.Linear(hidden_dim, output_dim))
    model = nn.Sequential(*layers)
    
    return model
