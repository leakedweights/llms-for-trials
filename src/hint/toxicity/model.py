import torch.nn as nn
import torch

class MultitaskToxicityModel(nn.Module):
    def __init__(self, input_shape, all_tasks, dropout_rate=0.5):
        super(MultitaskToxicityModel, self).__init__()
        
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
        
        self.embedding_size = len(all_tasks)

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
        y_pred = torch.stack(y_pred, dim=0)
        y_pred = y_pred.permute(1, 0, 2)
        return y_pred
    
def save_ckp(state, is_best, checkpoint_path, best_model_path):
    # Method from : https://gist.github.com/vsay01/45dfced69687077be53dbdd4987b6b17
    f_path = checkpoint_path
    torch.save(state, f_path)
    if is_best:
        best_fpath = best_model_path
        shutil.copyfile(f_path, best_fpath)
        
def load_ckp(checkpoint_fpath, input_model, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    input_model.load_state_dict(checkpoint['state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    train_loss_min = checkpoint['train_loss_min']
    return input_model, optimizer, checkpoint['epoch'], train_loss_min.item()