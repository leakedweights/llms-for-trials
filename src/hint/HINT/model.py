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



class Interaction(nn.Sequential):
    def __init__(self, molecule_encoder, disease_encoder, protocol_encoder, 
                    device, 
                    global_embed_size, 
                    highway_num_layer,
                    prefix_name, 
                    epoch = 20,
                    lr = 3e-4, 
                    weight_decay = 0,
                    ablations = {"config": {"base_model": True}}
                    ):
        super(Interaction, self).__init__()
        self.molecule_encoder = molecule_encoder 
        self.disease_encoder = disease_encoder 
        self.protocol_encoder = protocol_encoder 
        self.global_embed_size = global_embed_size
        self.ablations = ablations
        self.feature_dim = 0
        self.include_all = ablations["config"].get("base_model", False)
        
        self.feature_dim = self.molecule_encoder.embedding_size + self.disease_encoder.embedding_size + self.disease_encoder.embedding_size
            
        self.highway_num_layer = highway_num_layer 
        self.epoch = epoch 
        self.lr = lr 
        self.weight_decay = weight_decay 
        self.save_name = prefix_name + '_interaction'

        self.f = F.relu
        self.loss = nn.BCEWithLogitsLoss()
 
        self.encoder2interaction_fc = nn.Linear(self.feature_dim, self.global_embed_size).to(device)
        self.encoder2interaction_highway = Highway(self.global_embed_size, self.highway_num_layer).to(device)
        self.pred_nn = nn.Linear(self.global_embed_size, 1)

        self.device = device 
        self = self.to(device)

    def feed_lst_of_module(self, input_feature, lst_of_module):
        x = input_feature
        for single_module in lst_of_module:
            x = self.f(single_module(x))
        return x

    def forward_get_three_encoders(self, smiles_lst2, icdcode_lst3, criteria_lst):
        molecule_embed = self.molecule_encoder.forward_smiles_lst_lst(smiles_lst2)
        icd_embed = self.disease_encoder.forward_code_lst3(icdcode_lst3)
        protocol_embed = self.protocol_encoder.forward(criteria_lst)
        return molecule_embed, icd_embed, protocol_embed    

    def forward_encoder_2_interaction(self, molecule_embed, icd_embed, protocol_embed):
        encoder_embedding = torch.cat([molecule_embed, icd_embed, protocol_embed], 1)
        h = self.encoder2interaction_fc(encoder_embedding)
        h = self.f(h)
        h = self.encoder2interaction_highway(h)
        interaction_embedding = self.f(h)
        return interaction_embedding 

    def forward(self, smiles_lst2, icdcode_lst3, criteria_lst):
        molecule_embed, icd_embed, protocol_embed = self.forward_get_three_encoders(smiles_lst2, icdcode_lst3, criteria_lst)
        interaction_embedding = self.forward_encoder_2_interaction(molecule_embed, icd_embed, protocol_embed)
        output = self.pred_nn(interaction_embedding)
        return output

    def evaluation(self, predict_all, label_all, threshold = 0.5):
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

    def bootstrap_test(self, dataloader, sample_num = 20):
        self.eval()
        best_threshold = 0.5 
        whole_loss, predict_all, label_all, nctid_all = self.generate_predict(dataloader)
        from HINT.utils import plot_hist
        import matplotlib.pyplot as plt
        import numpy as np
        import pickle
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
        accuracy = [results[3] for results in results_lst]

        print("PR-AUC   mean: "+str(np.mean(prauc_score))[:6], "std: "+str(np.std(prauc_score))[:6])
        print("F1       mean: "+str(np.mean(f1score))[:6], "std: "+str(np.std(f1score))[:6])
        print("ROC-AUC  mean: "+ str(np.mean(auc))[:6], "std: " + str(np.std(auc))[:6])
        print("Accuracy mean: "+ str(np.mean(accuracy))[:6], "std: " + str(np.std(accuracy))[:6])
        
        nctid2predict = {nctid:predict for nctid, predict in zip(nctid_all, predict_all)} 
        pickle.dump(nctid2predict, open('results/nctid2predict.pkl', 'wb'))
        return nctid_all, predict_all

    def ongoing_test(self, dataloader, sample_num = 20):
        self.eval()
        best_threshold = 0.5 
        whole_loss, predict_all, label_all, nctid_all = self.generate_predict(dataloader) 
        self.train() 
        return nctid_all, predict_all

    def test(self, dataloader, return_loss = True, validloader=None):
        self.eval()
        best_threshold = 0.5 
        whole_loss, predict_all, label_all, nctid_all = self.generate_predict(dataloader)
        self.train()
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

    def learn(self, train_loader, valid_loader, test_loader):
        opt = torch.optim.Adam(self.parameters(), lr = self.lr, weight_decay = self.weight_decay)
        train_loss_record = [] 
        valid_loss = self.test(valid_loader, return_loss=True)
        valid_loss_record = [valid_loss]
        best_valid_loss = valid_loss
        best_model = deepcopy(self)
        for ep in tqdm(range(self.epoch)):
            for nctid_lst, label_vec, smiles_lst2, icdcode_lst3, criteria_lst in train_loader:
                label_vec = label_vec.to(self.device)
                output = self.forward(smiles_lst2, icdcode_lst3, criteria_lst).view(-1)
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

    def select_threshold_for_binary(self, validloader):
        _, prediction, label_all, nctid_all = self.generate_predict(validloader)
        best_f1 = 0
        for threshold in prediction:
            float2binary = lambda x:0 if x<threshold else 1
            predict_all = list(map(float2binary, prediction))
            f1score = precision_score(label_all, predict_all)        
            if f1score > best_f1:
                best_f1 = f1score 
                best_threshold = threshold
        return best_threshold 


class HINT_nograph(Interaction):
    def __init__(self, molecule_encoder, disease_encoder, protocol_encoder, device, 
                    global_embed_size, 
                    highway_num_layer,
                    prefix_name, 
                    epoch = 20,
                    lr = 3e-4, 
                    weight_decay = 0,
                    ablations = {"config": {"base_model": True}}):
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
                                   ablations=ablations,
                                   ) 
        self.save_name = prefix_name + '_HINT_nograph'


        self.risk_disease_fc = nn.Linear(self.disease_encoder.embedding_size, self.global_embed_size)
        self.risk_disease_higway = Highway(self.global_embed_size, self.highway_num_layer)
            
        self.augment_interaction_fc = nn.Linear(self.global_embed_size * 2, self.global_embed_size)
        self.augment_interaction_highway = Highway(self.global_embed_size, self.highway_num_layer)
    

        self.admet_model = []
        for i in range(5):
            admet_fc = nn.Linear(self.molecule_encoder.embedding_size, self.global_embed_size).to(device)
            admet_highway = Highway(self.global_embed_size, self.highway_num_layer).to(device)
            self.admet_model.append(nn.ModuleList([admet_fc, admet_highway])) 
        self.admet_model = nn.ModuleList(self.admet_model)

        self.pk_fc = nn.Linear(self.global_embed_size*5, self.global_embed_size)
        self.pk_highway = Highway(self.global_embed_size, self.highway_num_layer)
            
        self.trial_fc = nn.Linear(self.global_embed_size * 2, self.global_embed_size)
        self.trial_highway = Highway(self.global_embed_size, self.highway_num_layer)

        self.device = device 
        self = self.to(device)


    def forward(self, smiles_lst2, icdcode_lst3, criteria_lst, if_gnn = False):

        molecule_embed, icd_embed, protocol_embed = self.forward_get_three_encoders(smiles_lst2, icdcode_lst3, criteria_lst)
        interaction_embedding = self.forward_encoder_2_interaction(molecule_embed, icd_embed, protocol_embed)
        risk_of_disease_embedding = self.feed_lst_of_module(input_feature = icd_embed, 
                                                            lst_of_module = [self.risk_disease_fc, self.risk_disease_higway])
        
        augment_interaction_input = torch.cat([interaction_embedding, risk_of_disease_embedding], 1)
        augment_interaction_embedding = self.feed_lst_of_module(input_feature = augment_interaction_input, 
                                                                lst_of_module = [self.augment_interaction_fc, self.augment_interaction_highway])
        
        admet_embedding_lst = []
        for idx in range(5):
            admet_embedding = self.feed_lst_of_module(input_feature = molecule_embed, 
                                                      lst_of_module = self.admet_model[idx])
            admet_embedding_lst.append(admet_embedding)

        pk_input = torch.cat(admet_embedding_lst, 1)
        pk_embedding = self.feed_lst_of_module(input_feature = pk_input, lst_of_module = [self.pk_fc, self.pk_highway])
        
        trial_input = torch.cat([augment_interaction_embedding, pk_embedding], 1)
        trial_embedding = self.feed_lst_of_module(input_feature = trial_input, 
                                              lst_of_module = [self.trial_fc, self.trial_highway])
        
        embedding_options = {
            "disease_embedding": icd_embed,
            "molecule_embedding": molecule_embed,
            "protocol_embedding": protocol_embed,
            "interaction_embedding": interaction_embedding,
            "disease_risk_embedding": risk_of_disease_embedding,
            "augmented_interaction_embedding": augment_interaction_embedding,
            "pk_embedding": pk_embedding,
            "trial_embedding": trial_embedding
        }

        for key, value in embedding_options.items():
            if self.ablations["config"].get(key, self.include_all):
                x = value
        
        if not self.include_all:
            output = self.pred_nn(x)
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
                    weight_decay = 0,
                    ablations = {"config": {"base_model": True}}):
        super(HINTModel, self).__init__(molecule_encoder = molecule_encoder, 
                                   disease_encoder = disease_encoder, 
                                   protocol_encoder = protocol_encoder, 
                                   device = device, 
                                   prefix_name = prefix_name, 
                                   global_embed_size = global_embed_size, 
                                   highway_num_layer = highway_num_layer,
                                   epoch = epoch,
                                   lr = lr, 
                                   weight_decay = weight_decay,
                                   ablations=ablations)
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
        ### gnn's attention         
        self.node_size = self.adj.shape[0]

        self.graph_attention_model_mat = nn.ModuleList([nn.ModuleList([self.gnn_attention() if self.adj[i,j]==1 else None for j in range(self.node_size)]) for i in range(self.node_size)])


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
                attention_mat[i,j] = torch.sigmoid(self.feed_lst_of_module(input_feature=feature, lst_of_module=attention_model))
        return attention_mat 

    def gnn_attention(self):
        highway_nn = Highway(size = self.global_embed_size*2, num_layers = self.highway_num_layer).to(self.device)
        highway_fc = nn.Linear(self.global_embed_size*2, 1).to(self.device)
        return nn.ModuleList([highway_nn, highway_fc])    

    def forward(self, smiles_lst2, icdcode_lst3, criteria_lst, return_attention_matrix = False):
        
        if not self.ablations.get("gnn", self.include_all):
            return HINT_nograph.forward(self, smiles_lst2, icdcode_lst3, criteria_lst, if_gnn = False)
        
        embedding_lst = HINT_nograph.forward(self, smiles_lst2, icdcode_lst3, criteria_lst, if_gnn = True)
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
        for nctid_lst, status_lst, why_stop_lst, label_vec, phase_lst, \
            diseases_lst, icdcode_lst3, drugs_lst, smiles_lst2, criteria_lst in complete_dataloader: 
            output, attention_mat_lst = self.forward(smiles_lst2, icdcode_lst3, criteria_lst, return_attention_matrix=True)
            output = output.view(-1)
            batch_size = len(nctid_lst)
            for i in range(batch_size):
                name = '__'.join([nctid_lst[i], status_lst[i], why_stop_lst[i], \
                                                        str(label_vec[i].item()), str(torch.sigmoid(output[i]).item())[:5], \
                                                        phase_lst[i], diseases_lst[i], drugs_lst[i]])
                if len(name) > 150:
                    name = name[:250]
                name = replace_strange_symbol(name)
                name = name.replace('__', '_')
                name = name.replace('  ', ' ')
                name = 'interpret_result/' + name + '.png'
                data2graph(attention_matrix = attention_mat_lst[i], adj = self.adj, save_name = name)

    def init_pretrain(self, admet_model):
        self.molecule_encoder = admet_model.molecule_encoder





