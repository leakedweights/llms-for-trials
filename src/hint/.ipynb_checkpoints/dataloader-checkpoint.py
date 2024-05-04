import torch, csv, os
from torch.utils import data 
from torch.utils.data.dataloader import default_collate
from HINT.molecule_encode import smiles2mpnnfeature
from HINT.protocol_embedding import protocol2feature, load_sentence_2_vec

sentence2vec = load_sentence_2_vec('data/bio_clinicalbert_embeddings.pkl') 

class TrialDataset(data.Dataset):
    def __init__(self, nctid_lst, label_lst, smiles_lst, icdcode_lst, criteria_lst):
        self.nctid_lst = nctid_lst 
        self.label_lst = label_lst 
        self.smiles_lst = smiles_lst 
        self.icdcode_lst = icdcode_lst 
        self.criteria_lst = criteria_lst 

    def __len__(self):
        return len(self.nctid_lst)

    def __getitem__(self, index):
        return self.nctid_lst[index], self.label_lst[index], self.smiles_lst[index], self.icdcode_lst[index], self.criteria_lst[index]


class ADMETDataset(data.Dataset):
    def __init__(self, smiles_lst, label_lst):
        self.smiles_lst = smiles_lst 
        self.label_lst = label_lst 
    
    def __len__(self):
        return len(self.smiles_lst)

    def __getitem__(self, index):
        return self.smiles_lst[index], self.label_lst[index]

def admet_collate_fn(x):
    smiles_lst = [i[0] for i in x]
    label_vec = default_collate([int(i[1]) for i in x])
    return [smiles_lst, label_vec]


def smiles_txt_to_lst(text):
    text = text[1:-1]
    lst = [i.strip()[1:-1] for i in text.split(',')]
    return lst 

def icdcode_text_2_lst_of_lst(text):
    text = text[2:-2]
    lst_lst = []
    for i in text.split('", "'):
        i = i[1:-1]
        lst_lst.append([j.strip()[1:-1] for j in i.split(',')])
    return lst_lst 

def trial_collate_fn(x):
    nctid_lst = [i[0] for i in x]
    label_vec = default_collate([int(i[1]) for i in x])
    smiles_lst = [smiles_txt_to_lst(i[2]) for i in x]
    icdcode_lst = [icdcode_text_2_lst_of_lst(i[3]) for i in x]
    criteria_lst = [protocol2feature(i[4], sentence2vec) for i in x]
    return [nctid_lst, label_vec, smiles_lst, icdcode_lst, criteria_lst]

def csv_three_feature_2_dataloader(csvfile, shuffle, batch_size):
    with open(csvfile, 'r') as csvfile:
        rows = list(csv.reader(csvfile, delimiter=','))[1:]
    nctid_lst = [row[0] for row in rows]
    label_lst = [row[3] for row in rows]
    icdcode_lst = [row[6] for row in rows]
    drugs_lst = [row[7] for row in rows]
    smiles_lst = [row[8] for row in rows]
    criteria_lst = [row[9] for row in rows] 
    dataset = TrialDataset(nctid_lst, label_lst, smiles_lst, icdcode_lst, criteria_lst)
    data_loader = data.DataLoader(dataset, batch_size = batch_size, shuffle = shuffle, collate_fn = trial_collate_fn)
    return data_loader


def smiles_txt_to_2lst(smiles_txt_file):
    with open(smiles_txt_file, 'r') as fin:
        lines = fin.readlines() 
    smiles_lst = [line.split()[0] for line in lines]
    label_lst = [int(line.split()[1]) for line in lines]
    return smiles_lst, label_lst 

def generate_admet_dataloader_lst(batch_size):
    datafolder = "data/ADMET/cooked/"
    name_lst = ["absorption", 'distribution', 'metabolism', 'excretion', 'toxicity']
    dataloader_lst = []
    for i,name in enumerate(name_lst):
        train_file = os.path.join(datafolder, name + '_train.txt')
        test_file = os.path.join(datafolder, name +'_valid.txt')
        train_smiles_lst, train_label_lst = smiles_txt_to_2lst(train_file)
        test_smiles_lst, test_label_lst = smiles_txt_to_2lst(test_file)
        train_dataset = ADMETDataset(smiles_lst = train_smiles_lst, label_lst = train_label_lst)
        test_dataset = ADMETDataset(smiles_lst = test_smiles_lst, label_lst = test_label_lst)
        train_dataloader = data.DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
        test_dataloader = data.DataLoader(test_dataset, batch_size = batch_size, shuffle = False)
        dataloader_lst.append((train_dataloader, test_dataloader))
    return dataloader_lst 




















