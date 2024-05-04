import csv, re, pickle, os 
from functools import reduce 
import icd10
from collections import defaultdict

import torch 
torch.manual_seed(0)
from torch import nn 
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils import data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def text_2_lst_of_lst(text):
    text = text[2:-2]
    code_sublst = []
    for i in text.split('", "'):
        i = i[1:-1]
        code_sublst.append([j.strip()[1:-1] for j in i.split(',')])
    # print(code_sublst)    
    return code_sublst 

def get_icdcode_lst():
    input_file = 'data/raw_data.csv'
    with open(input_file, 'r') as csvfile:
        rows = list(csv.reader(csvfile, delimiter = ','))[1:]
    code_lst = []
    for row in rows:
        code_sublst = text_2_lst_of_lst(row[6])
        code_lst.append(code_sublst)
    return code_lst 

def combine_lst_of_lst(lst_of_lst):
    lst = list(reduce(lambda x,y:x+y, lst_of_lst))
    lst = list(set(lst))
    return lst

def collect_all_icdcodes():
    code_lst = get_icdcode_lst()
    code_lst = list(map(combine_lst_of_lst, code_lst))
    code_lst = list(reduce(lambda x,y:x+y, code_lst))
    code_lst = list(set(code_lst))
    return code_lst


def find_ancestor_for_icdcode(icdcode, icdcode2ancestor):
    if icdcode in icdcode2ancestor:
        return 
    icdcode2ancestor[icdcode] = []
    ancestor = icdcode[:]
    while len(ancestor) > 2:
        ancestor = ancestor[:-1]
        if ancestor[-1]=='.':
            ancestor = ancestor[:-1]
        if icd10.find(ancestor) is not None:
            icdcode2ancestor[icdcode].append(ancestor)
    return


def build_icdcode2ancestor_dict():
    pkl_file = "data/icdcode2ancestor_dict.pkl"
    if os.path.exists(pkl_file):
        icdcode2ancestor = pickle.load(open(pkl_file, 'rb'))
        return icdcode2ancestor 
    all_code = collect_all_icdcodes() 
    icdcode2ancestor = defaultdict(list)
    for code in all_code:
        find_ancestor_for_icdcode(code, icdcode2ancestor)
    pickle.dump(icdcode2ancestor, open(pkl_file,'wb'))
    return icdcode2ancestor 


def collect_all_code_and_ancestor():
    icdcode2ancestor = build_icdcode2ancestor_dict()
    all_code = set(icdcode2ancestor.keys())
    ancestor_lst = list(icdcode2ancestor.values())
    ancestor_set = set(reduce(lambda x,y:x+y, ancestor_lst))
    all_code_lst = all_code.union(ancestor_set)
    return all_code_lst    


class GRAM(nn.Sequential):
    def __init__(self, embedding_dim, icdcode2ancestor, device):
        super(GRAM, self).__init__()        
        self.icdcode2ancestor = icdcode2ancestor 
        self.all_code_lst = GRAM.codedict_2_allcode(self.icdcode2ancestor)
        self.code_num = len(self.all_code_lst)
        self.maxlength = 5
        self.code2index = {code:idx for idx,code in enumerate(self.all_code_lst)}
        self.index2code = {idx:code for idx,code in enumerate(self.all_code_lst)}
        self.padding_matrix = torch.zeros(self.code_num, self.maxlength).long() 
        self.mask_matrix = torch.zeros(self.code_num, self.maxlength)
        for idx in range(self.code_num):
            code = self.index2code[idx]
            ancestor_code_lst = self.icdcode2ancestor[code]
            ancestor_idx_lst = [idx] + [self.code2index[code] for code in ancestor_code_lst]
            ancestor_mask_lst = [1 for i in ancestor_idx_lst] + [0] * (self.maxlength - len(ancestor_idx_lst))
            ancestor_idx_lst = ancestor_idx_lst + [0]*(self.maxlength-len(ancestor_idx_lst))
            self.padding_matrix[idx,:] = torch.Tensor(ancestor_idx_lst)
            self.mask_matrix[idx,:] = torch.Tensor(ancestor_mask_lst)

        self.embedding_dim = embedding_dim 
        self.embedding = nn.Embedding(self.code_num, self.embedding_dim)
        self.attention_model = nn.Linear(2*embedding_dim, 1)

        self.device = device
        self = self.to(device)
        self.padding_matrix = self.padding_matrix.to('cpu')
        self.mask_matrix = self.mask_matrix.to('cpu')

    @property
    def embedding_size(self):
        return self.embedding_dim


    @staticmethod
    def codedict_2_allcode(icdcode2ancestor):
        all_code = set(icdcode2ancestor.keys())
        ancestor_lst = list(icdcode2ancestor.values())
        ancestor_set = set(reduce(lambda x,y:x+y, ancestor_lst))
        all_code_lst = all_code.union(ancestor_set)
        return all_code_lst        


    def forward_single_code(self, single_code):
        idx = self.code2index[single_code].to(self.device)
        ancestor_vec = self.padding_matrix[idx,:]
        mask_vec = self.mask_matrix[idx,:] 

        embeded = self.embedding(ancestor_vec)
        current_vec = torch.cat([self.embedding(torch.Tensor([idx]).long()).view(1,-1) for i in range(self.maxlength)], 0)
        attention_input = torch.cat([embeded, current_vec], 1)
        attention_weight = self.attention_model(attention_input)
        attention_weight = torch.exp(attention_weight)
        attention_output = attention_weight * mask_vec.view(-1,1)
        attention_output = attention_output / torch.sum(attention_output)
        output = embeded * attention_output
        output = torch.sum(output, 0)
        return output 


    def forward_code_lst(self, code_lst):
        idx_lst = [self.code2index[code] for code in code_lst if code in self.code2index]
        if idx_lst == []:
            idx_lst = [0]
        ancestor_mat = self.padding_matrix[idx_lst,:].to(self.device)
        mask_mat = self.mask_matrix[idx_lst,:].to(self.device)
        embeded = self.embedding(ancestor_mat)
        current_vec = self.embedding(torch.Tensor(idx_lst).long().to(self.device))
        current_vec = current_vec.unsqueeze(1)
        current_vec = current_vec.repeat(1, self.maxlength, 1)
        attention_input = torch.cat([embeded, current_vec], 2)
        attention_weight = self.attention_model(attention_input)
        attention_weight = torch.exp(attention_weight).squeeze(-1)
        attention_output = attention_weight * mask_mat
        attention_output = attention_output / torch.sum(attention_output, 1).view(-1,1)
        attention_output = attention_output.unsqueeze(-1)
        output = embeded * attention_output
        output = torch.sum(output,1)
        return output 

    def forward_code_lst2(self, code_lst_lst):
        code_lst = reduce(lambda x,y:x+y, code_lst_lst)
        code_embed = self.forward_code_lst(code_lst)
        code_embed = torch.mean(code_embed, 0).view(1,-1)
        return code_embed 
        
    def forward_code_lst3(self, code_lst_lst_lst):
        code_embed_lst = [self.forward_code_lst2(code_lst_lst) for code_lst_lst in code_lst_lst_lst]
        code_embed = torch.cat(code_embed_lst, 0)
        return code_embed 





if __name__ == '__main__':
    dic = build_icdcode2ancestor_dict()




