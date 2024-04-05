from sklearn.metrics import roc_auc_score, f1_score, average_precision_score, precision_score, recall_score, accuracy_score
import matplotlib.pyplot as plt
from copy import deepcopy 
import numpy as np 
from tqdm import tqdm 
import torch 
torch.manual_seed(0)
from torch import nn 
from torch.autograd import Variable
import torch.nn.functional as F
from HINT.module import Highway, GCN 
from functools import reduce 
import pickle

from dataclasses import dataclass

@dataclass
class HintModelConfig:
    
    global_embed_size: int
    num_highway_layers: int
    
    molecule_encoder: nn.Module
    disease_encoder: nn.Module
    protocol_encoder:nn.Module
    
    model_name: str
    
class Interaction(nn.Module):
    def __init__(self, config: HintModelConfig):
        self.config = config
        
        self.embedding_interactions = nn.Sequential(nn.Linear(config.feature_dim, self.global_embed_size),
                                                 nn.ReLU(),
                                                 Highway(config.global_embed_size, self.highway_num_layer),
                                                 nn.ReLU())
        
        self.out_proj = nn.Linear(self.global_embed_size, 1)
        
    def forward(self, smiles, icd_codes, criteria):
        config = self.config
        
        molecule_embedding = config.molecule_encoder.forward_smiles_lst_lst(smiles)
        icd_embedding = config.disease_encoder.forward_code_lst3(icdcodes)
        protocol_embedding = config.protocol_encoder.forward(criteria)
        
        embeddings = torch.cat([molecule_embedding, icd_embedding, protocol_embedding], 1)
        
        interaction_embedding = config.embedding_interactions(embeddings)
        output = self.out_proj(interaction_embedding)
        return output

# ------------ old util fns

def evaluate(predict_all, label_all, threshold = 0.5):
        import pickle, os
        from sklearn.metrics import roc_curve, precision_recall_curve
        with open("predict_label.txt", 'w') as fout:
            for i,j in zip(predict_all, label_all):
                fout.write(str(i)[:4] + '\t' + str(j)[:4]+'\n')
        auc_score = roc_auc_score(label_all, predict_all)
        figure_folder = "figure"

        fpr, tpr, thresholds = roc_curve(label_all, predict_all, pos_label=1)

        precision, recall, thresholds = precision_recall_curve(label_all, predict_all)

        label_all = [int(i) for i in label_all]
        float2binary = lambda x:0 if x<threshold else 1
        predict_all = list(map(float2binary, predict_all))
        f1score = f1_score(label_all, predict_all)
        prauc_score = average_precision_score(label_all, predict_all)

        precision = precision_score(label_all, predict_all)
        recall = recall_score(label_all, predict_all)
        accuracy = accuracy_score(label_all, predict_all)
        predict_1_ratio = sum(predict_all) / len(predict_all)
        label_1_ratio = sum(label_all) / len(label_all)
        return auc_score, f1score, prauc_score, precision, recall, accuracy, predict_1_ratio, label_1_ratio
    
def testloader_to_lst(self, dataloader):
        nctid_lst, label_lst, smiles_lst2, icdcode_lst3, criteria_lst = [], [], [], [], []
        for nctid, label, smiles, icdcode, criteria in dataloader:
            nctid_lst.extend(nctid)
            label_lst.extend([i.item() for i in label])
            smiles_lst2.extend(smiles)
            icdcode_lst3.extend(icdcode)
            criteria_lst.extend(criteria)
        length = len(nctid_lst)
        assert length == len(smiles_lst2) and length == len(icdcode_lst3)
        return nctid_lst, label_lst, smiles_lst2, icdcode_lst3, criteria_lst, length
    
def bootstrap_test(model, dataloader, sample_num = 20):
        model.eval()
        best_threshold = 0.5 
        whole_loss, predict_all, label_all, nctid_all = model.generate_predict(dataloader)
        from HINT.utils import plot_hist
        plt.clf()
        prefix_name = "./figure/" + self.save_name 
        plot_hist(prefix_name, predict_all, label_all)        
        def bootstrap(length, sample_num):
            idx = [i for i in range(length)]
            from random import choices 
            bootstrap_idx = [choices(idx, k = length) for i in range(sample_num)]
            return bootstrap_idx 
        results_lst = []
        bootstrap_idx_lst = bootstrap(len(predict_all), sample_num = sample_num)
        for bootstrap_idx in bootstrap_idx_lst: 
            bootstrap_label = [label_all[idx] for idx in bootstrap_idx]        
            bootstrap_predict = [predict_all[idx] for idx in bootstrap_idx]
            results = self.evaluation(bootstrap_predict, bootstrap_label, threshold = best_threshold)
            results_lst.append(results)
        self.train() 
        auc = [results[0] for results in results_lst]
        f1score = [results[1] for results in results_lst]
        prauc_score = [results[2] for results in results_lst]
        print("PR-AUC   mean: "+str(np.mean(prauc_score))[:6], "std: "+str(np.std(prauc_score))[:6])
        print("F1       mean: "+str(np.mean(f1score))[:6], "std: "+str(np.std(f1score))[:6])
        print("ROC-AUC  mean: "+ str(np.mean(auc))[:6], "std: " + str(np.std(auc))[:6])

        for nctid, label, predict in zip(nctid_all, label_all, predict_all):
            if (predict > 0.5 and label == 0) or (predict < 0.5 and label == 1):
                print(nctid, label, str(predict)[:5])

        nctid2predict = {nctid:predict for nctid, predict in zip(nctid_all, predict_all)} 
        pickle.dump(nctid2predict, open('results/nctid2predict.pkl', 'wb'))
        return nctid_all, predict_all 
    
def test(model, dataloader, return_loss = True, validloader=None):
        model.eval()
        best_threshold = 0.5 
        whole_loss, predict_all, label_all, nctid_all = self.generate_predict(dataloader)
        model.train()
        if return_loss:
            return whole_loss
        else:
            print_num = 5
            auc_score, f1score, prauc_score, precision, recall, accuracy, \
            predict_1_ratio, label_1_ratio = self.evaluation(predict_all, label_all, threshold = best_threshold)
            print("ROC AUC: " + str(auc_score)[:print_num] + "\nF1: " + str(f1score)[:print_num] \
                 + "\nPR-AUC: " + str(prauc_score)[:print_num] \
                 + "\nPrecision: " + str(precision)[:print_num] \
                 + "\nrecall: "+str(recall)[:print_num] + "\naccuracy: "+str(accuracy)[:print_num] \
                 + "\npredict 1 ratio: " + str(predict_1_ratio)[:print_num] \
                 + "\nlabel 1 ratio: " + str(label_1_ratio)[:print_num])
            return auc_score, f1score, prauc_score, precision, recall, accuracy, predict_1_ratio, label_1_ratio
        
def select_threshold_for_binary(model, validloader):
        _, prediction, label_all, nctid_all = self.model(validloader)
        best_f1 = 0
        for threshold in prediction:
            float2binary = lambda x:0 if x<threshold else 1
            predict_all = list(map(float2binary, prediction))
            f1score = precision_score(label_all, predict_all)        
            if f1score > best_f1:
                best_f1 = f1score 
                best_threshold = threshold
        return best_threshold 

# ------------ old models
    
class Interaction(nn.Sequential):
    def __init__(self, molecule_encoder, disease_encoder, protocol_encoder, 
                    device, 
                    global_embed_size, 
                    highway_num_layer,
                    prefix_name, 
                    epoch = 20,
                    lr = 3e-4, 
                    weight_decay = 0, 
                    ):
        super(Interaction, self).__init__()
        self.molecule_encoder = molecule_encoder 
        self.disease_encoder = disease_encoder 
        self.protocol_encoder = protocol_encoder 
        self.global_embed_size = global_embed_size 
        self.highway_num_layer = highway_num_layer 
        self.feature_dim = self.molecule_encoder.embedding_size + self.disease_encoder.embedding_size + self.protocol_encoder.embedding_size
        self.epoch = epoch 
        self.lr = lr 
        self.weight_decay = weight_decay 
        self.save_name = prefix_name + '_interaction'

        self.f = F.relu
        self.loss = nn.BCEWithLogitsLoss()
        
        self.encoder2interaction = nn.Sequential(nn.Linear(self.feature_dim, self.global_embed_size),
                                                 nn.ReLU(),
                                                 Highway(self.global_embed_size, self.highway_num_layer),
                                                 nn.ReLU())

        self.pred_nn = nn.Linear(self.global_embed_size, 1)

    def forward(self, smiles, icdcodes, criteria):
        molecule_embedding = self.molecule_encoder.forward_smiles_lst_lst(smiles)
        icd_embedding = self.disease_encoder.forward_code_lst3(icdcodes)
        protocol_embedding = self.protocol_encoder.forward(criteria)
        
        embeddings = torch.cat([molecule_embedding, icd_embedding, protocol_embedding], 1)
        
        interaction_embedding = self.encoder2interaction(embeddings)
        output = self.pred_nn(interaction_embedding)
        return output 

    def generate_predict(self, dataloader):
        whole_loss = 0 
        label_all, predict_all, nctid_all = [], [], []
        for nctid_lst, label_vec, smiles_lst2, icdcode_lst3, criteria_lst in dataloader:
            nctid_all.extend(nctid_lst)
            label_vec = label_vec.to(self.device)
            output = self.forward(smiles_lst2, icdcode_lst3, criteria_lst).view(-1)  
            loss = self.loss(output, label_vec.float())
            whole_loss += loss.item()
            predict_all.extend([i.item() for i in torch.sigmoid(output)])
            label_all.extend([i.item() for i in label_vec])

        return whole_loss, predict_all, label_all, nctid_all

    def learn(self, train_loader, valid_loader, test_loader):
        opt = torch.optim.Adam(self.parameters(), lr = self.lr, weight_decay = self.weight_decay)
        train_loss_record = [] 
        valid_loss = self.test(valid_loader, return_loss=True)
        valid_loss_record = [valid_loss]
        best_valid_loss = valid_loss
        best_model = deepcopy(self)
        for ep in tqdm(range(self.epoch)):
            for nctid_lst, label_vec, smiles, icdcodes, criteria in train_loader:
                label_vec = label_vec.to(self.device)
                output = self.forward(smiles, icdcodes, criteria).view(-1)
                loss = self.loss(output, label_vec.float())
                train_loss_record.append(loss.item())
                opt.zero_grad() 
                loss.backward() 
                opt.step()
            valid_loss = self.test(valid_loader, return_loss=True)
            valid_loss_record.append(valid_loss)
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss 
                best_model = deepcopy(self)

        self.plot_learning_curve(train_loss_record, valid_loss_record)
        self = deepcopy(best_model)
        auc_score, f1score, prauc_score, precision, recall, accuracy, predict_1_ratio, label_1_ratio = self.test(test_loader, return_loss = False, validloader = valid_loader)


    def plot_learning_curve(self, train_loss_record, valid_loss_record):
        plt.plot(train_loss_record)
        plt.savefig("./figure/" + self.save_name + '_train_loss.jpg')
        plt.clf() 
        plt.plot(valid_loss_record)
        plt.savefig("./figure/" + self.save_name + '_valid_loss.jpg')
        plt.clf() 


class HINT_nograph(Interaction):
    def __init__(self, molecule_encoder, disease_encoder, protocol_encoder, device, 
                    global_embed_size, 
                    highway_num_layer,
                    prefix_name, 
                    epoch = 20,
                    lr = 3e-4, 
                    weight_decay = 0, ):
        super(HINT_nograph, self).__init__(molecule_encoder = molecule_encoder, 
                                   disease_encoder = disease_encoder, 
                                   protocol_encoder = protocol_encoder,
                                   device = device,  
                                   global_embed_size = global_embed_size, 
                                   prefix_name = prefix_name, 
                                   highway_num_layer = highway_num_layer,
                                   epoch = epoch,
                                   lr = lr, 
                                   weight_decay = weight_decay, 
                                   ) 
        self.save_name = prefix_name + '_HINT_nograph'
        
        self.disease_risk = nn.Sequential(nn.Linear(self.disease_encoder.embedding_size, self.global_embed_size),
                                            nn.ReLU(),
                                            Highway(self.global_embed_size, self.highway_num_layer))
        
        self.interaction = nn.Sequential(nn.Linear(self.global_embed_size*2, self.global_embed_size),
                                                nn.ReLU(),
                                                Highway(self.global_embed_size, self.highway_num_layer))
                                                
        self.pharmacokinetics = nn.Sequential(nn.Linear(self.global_embed_size*5, self.global_embed_size),
                                                nn.ReLU(),
                                                Highway(self.global_embed_size, self.highway_num_layer))
        
        self.trial = nn.Sequential(nn.Linear(self.global_embed_size*2, self.global_embed_size),
                                          nn.ReLU(),
                                          Highway(self.global_embed_size, self.highway_num_layer))
        
        self.admet = nn.ModuleList(nn.Sequential(
            nn.Linear(self.molecule_encoder.embedding_size, self.global_embed_size),
            nn.ReLU(),
            Highway(self.global_embed_size, self.highway_num_layer),
            nn.ReLU()
        ) for admet_component in "ADMET")


    def forward(self, smiles, icdcodes, criteria, return_embeddings = False):
        molecule_embedding = self.molecule_encoder.forward_smiles_lst_lst(smiles)
        icd_embedding = self.disease_encoder.forward_code_lst3(icdcodes)
        protocol_embedding = self.protocol_encoder.forward(criteria)
        disease_embedding = self.disease_risk(icd_embedding)
        
        interaction_embedding = self.encoder2interaction(torch.cat([molecule_embedding, icd_embedding, protocol_embedding], 1))
        
        augment_interaction_input = torch.cat([interaction_embedding, disease_embedding], 1)
        augment_interaction_embedding = self.interaction(augment_interaction_input)
                                                        
        admet_embeddings = [admet_component(molecule_embed) for admet_component in self.admet]
                                                        
        pk_input = torch.cat(admet_embedding_lst, 1)
        pk_embedding = self.pharmacokinetics(pk_input)
                                                        
        trial_input = torch.cat([pk_embedding, augment_interaction_embedding], 1)
        trial_embedding = self.trial(trial_input)
                                     
        
        if not return_embeddings:
            output = self.pred_nn(trial_embedding)
            return output 
        else:
            embedding_lst = [molecule_embed, icd_embed, protocol_embed, interaction_embedding, risk_of_disease_embedding, \
                             augment_interaction_embedding] + admet_embedding_lst + [pk_embedding, trial_embedding]
            return embedding_lst

class HINTModel(HINT_nograph):

    def __init__(self, molecule_encoder, disease_encoder, protocol_encoder, 
                    device, 
                    global_embed_size, 
                    highway_num_layer,
                    prefix_name, 
                    gnn_hidden_size, 
                    epoch = 20,
                    lr = 3e-4, 
                    weight_decay = 0,):
        super(HINTModel, self).__init__(molecule_encoder = molecule_encoder, 
                                   disease_encoder = disease_encoder, 
                                   protocol_encoder = protocol_encoder, 
                                   device = device, 
                                   prefix_name = prefix_name, 
                                   global_embed_size = global_embed_size, 
                                   highway_num_layer = highway_num_layer,
                                   epoch = epoch,
                                   lr = lr, 
                                   weight_decay = weight_decay)
        self.save_name = prefix_name 
        self.gnn_hidden_size = gnn_hidden_size 
        #### GNN 
        self.adj = self.generate_adj()          
        self.gnn = GCN(
            nfeat = self.global_embed_size,
            nhid = self.gnn_hidden_size,
            nclass = 1,
            dropout = 0.6,
            init = 'uniform') 
     
        self.node_size = self.adj.shape[0]
        self.graph_attention_model_mat = nn.ModuleList([
            nn.ModuleList(
                [self.gnn_attention() if self.adj[i,j]==1 else None for j in range(self.node_size)]) for i in range(self.node_size)
        ])

        self.device = device 
        self = self.to(device)

    def generate_adj(self):                                        

        lst = ["molecule", "disease", "criteria", 'INTERACTION', 'risk_disease', 'augment_interaction', 'A', 'D', 'M', 'E', 'T', 'PK', "final"]
        edge_lst = [("disease", "molecule"), ("disease", "criteria"), ("molecule", "criteria"), 
                    ("disease", "INTERACTION"), ("molecule", "INTERACTION"),  ("criteria", "INTERACTION"), 
                    ("disease", "risk_disease"), ('risk_disease', 'augment_interaction'), ('INTERACTION', 'augment_interaction'),
                    ("molecule", "A"), ("molecule", "D"), ("molecule", "M"), ("molecule", "E"), ("molecule", "T"),
                    ('A', 'PK'), ('D', 'PK'), ('M', 'PK'), ('E', 'PK'), ('T', 'PK'), 
                    ('augment_interaction', 'final'), ('PK', 'final')]
        adj = torch.zeros(len(lst), len(lst))
        adj = torch.eye(len(lst)) * len(lst)
        num2str = {k:v for k,v in enumerate(lst)}
        str2num = {v:k for k,v in enumerate(lst)}
        for i,j in edge_lst:
            n1,n2 = str2num[i], str2num[j]
            adj[n1,n2] = 1
            adj[n2,n1] = 1
        return adj.to(self.device) 

    def generate_attention_matrx(self, node_feature_mat):
        attention_mat = torch.zeros(self.node_size, self.node_size).to(self.device)
        for i in range(self.node_size):
            for j in range(self.node_size):
                if self.adj[i,j]!=1:
                    continue 
                feature = torch.cat([node_feature_mat[i].view(1,-1), node_feature_mat[j].view(1,-1)], 1)
                attention_model = self.graph_attention_model_mat[i][j]
                attention_mat[i,j] = torch.sigmoid(self.stack_relu(input_feature=feature, lst_of_module=attention_model))
        return attention_mat 

    def gnn_attention(self):
        highway_nn = Highway(size = self.global_embed_size*2, num_layers = self.highway_num_layer).to(self.device)
        highway_fc = nn.Linear(self.global_embed_size*2, 1).to(self.device)
        return nn.ModuleList([highway_nn, highway_fc])    

    def forward(self, smiles_lst2, icdcode_lst3, criteria_lst, return_attention_matrix = False):
        embedding_lst = HINT_nograph.forward(self, smiles_lst2, icdcode_lst3, criteria_lst, return_embeddings = True)

        batch_size = embedding_lst[0].shape[0]
        output_lst = []
        if return_attention_matrix:
            attention_mat_lst = []
        for i in range(batch_size):
            node_feature_lst = [embedding[i].view(1,-1) for embedding in embedding_lst]
            node_feature_mat = torch.cat(node_feature_lst, 0)
            attention_mat = self.generate_attention_matrx(node_feature_mat)
            output = self.gnn(node_feature_mat, self.adj * attention_mat)
            output = output[-1].view(1,-1)
            output_lst.append(output)
            if return_attention_matrix:
                attention_mat_lst.append(attention_mat)
        output_mat = torch.cat(output_lst, 0)
        if not return_attention_matrix:
            return output_mat 
        else:
            return output_mat, attention_mat_lst

    def interpret(self, complete_dataloader):
        from graph_visualize_interpret import data2graph 
        from HINT.utils import replace_strange_symbol
        for nctids, statuses, why_stop_lst, label_vec, phases, \
            diseases, icdcodes, drugs, smiles, criteria in complete_dataloader: 
            output, attention_mat_lst = self.forward(smiles, icdcodes, criteria, return_attention_matrix=True)
            output = output.view(-1)
            batch_size = len(nctid_lst)
            for i in range(batch_size):
                name = '__'.join([nctids[i], statuses[i], why_stop_lst[i], \
                                                        str(label_vec[i].item()), str(torch.sigmoid(output[i]).item())[:5], \
                                                        phases[i], diseases_lst[i], drugs_lst[i]])
                if len(name) > 150:
                    name = name[:250]
                name = replace_strange_symbol(name)
                name = name.replace('__', '_')
                name = name.replace('  ', ' ')
                name = 'interpret_result/' + name + '.png'
                print(name)
                data2graph(attention_matrix = attention_mat_lst[i], adj = self.adj, save_name = name)

    def init_pretrain(self, admet_model):
        self.molecule_encoder = admet_model.molecule_encoder