import torch.nn as nn
import torch

class MTDNN(nn.Module):
    def __init__(self, input_shape, all_tasks, dropout_rate=0.5):
        super(MTDNN, self).__init__()
        
        self.shared_1 = nn.Linear(input_shape, 2048)
        self.batchnorm_1 = nn.BatchNorm1d(2048)
        self.dropout_1 = nn.Dropout(dropout_rate)
        
        self.shared_2 = nn.Linear(2048, 1024)
        self.batchnorm_2 = nn.BatchNorm1d(1024)
        self.dropout_2 = nn.Dropout(dropout_rate)
        
        self.hidden_3 = nn.ModuleList([nn.Linear(1024, 512) for task in all_tasks])
        self.batchnorm_3 = nn.ModuleList([nn.BatchNorm1d(512) for task in all_tasks])
        self.dropout_3 = nn.ModuleList([nn.Dropout(dropout_rate) for _ in all_tasks])
        
        self.hidden_4 = nn.ModuleList([nn.Linear(512, 256) for task in all_tasks])
        self.batchnorm_4 = nn.ModuleList([nn.BatchNorm1d(256) for task in all_tasks])
        self.dropout_4 = nn.ModuleList([nn.Dropout(dropout_rate) for _ in all_tasks])
        
        self.output = nn.ModuleList([nn.Linear(256, 1) for task in all_tasks])
        self.leakyReLU = nn.LeakyReLU(0.05)

    def forward(self, x):
        x = self.shared_1(x)
        x = self.batchnorm_1(x)
        x = self.leakyReLU(x)
        x = self.dropout_1(x)
        
        x = self.shared_2(x)
        x = self.batchnorm_2(x)
        x = self.leakyReLU(x)
        x = self.dropout_2(x)
        
        x_task = [None for i in range(len(self.output))]
        for task in range(len(self.output)):
            x_task[task] = self.hidden_3[task](x)
            x_task[task] = self.batchnorm_3[task](x_task[task])
            x_task[task] = self.leakyReLU(x_task[task])
            x_task[task] = self.dropout_3[task](x_task[task])
            
            x_task[task] = self.hidden_4[task](x_task[task])
            x_task[task] = self.batchnorm_4[task](x_task[task])
            x_task[task] = self.leakyReLU(x_task[task])
            x_task[task] = self.dropout_4[task](x_task[task])
            
            x_task[task] = self.output[task](x_task[task])
            x_task[task] = torch.sigmoid(x_task[task])
        
        y_pred = x_task
        
        return y_pred