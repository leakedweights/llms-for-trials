import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import pickle
from tqdm import tqdm

torch.manual_seed(0)

def get_all_protocols(filepath='data/raw_data.csv'):
    with open(filepath, 'r') as csvfile:
        return [row[9] for row in csv.reader(csvfile)][1:]

def clean_protocol(protocol):
    protocol = protocol.lower().split('\n')
    return [line.strip() for line in protocol if line.strip()]

def split_protocol(protocol):
    protocol_split = clean_protocol(protocol)
    inclusion_idx = exclusion_idx = len(protocol_split)
    
    for idx, sentence in enumerate(protocol_split):
        if "inclusion" in sentence: inclusion_idx = idx
        if "exclusion" in sentence: exclusion_idx = idx 
    
    if inclusion_idx < exclusion_idx < len(protocol_split):
        return protocol_split[inclusion_idx:exclusion_idx], protocol_split[exclusion_idx:]
    return (protocol_split,)

def protocol2feature(protocol, sentence_2_vec):
    result = split_protocol(protocol)
    inclusion_criteria, exclusion_criteria = result[0], result[-1]
    inclusion_feature = [sentence_2_vec[sentence].view(1,-1) for sentence in inclusion_criteria if sentence in sentence_2_vec]
    exclusion_feature = [sentence_2_vec[sentence].view(1,-1) for sentence in exclusion_criteria if sentence in sentence_2_vec]
    if inclusion_feature == []:
        inclusion_feature = torch.zeros(1,768)
    else:
        inclusion_feature = torch.cat(inclusion_feature, 0)
    if exclusion_feature == []:
        exclusion_feature = torch.zeros(1,768)
    else:
        exclusion_feature = torch.cat(exclusion_feature, 0)
    return inclusion_feature, exclusion_feature

class ProtocolEmbedding(nn.Sequential):
    def __init__(self, output_dim, highway_num, device ):
        super(ProtocolEmbedding, self).__init__()    
        self.input_dim = 768  
        self.output_dim = output_dim 
        self.highway_num = highway_num 
        self.fc = nn.Linear(self.input_dim*2, output_dim)
        self.f = F.relu
        self.device = device 
        self = self.to(device)

    def forward_single(self, inclusion_feature, exclusion_feature):
        ## inclusion_feature, exclusion_feature: xxx,768 
        inclusion_feature = inclusion_feature.to(self.device)
        exclusion_feature = exclusion_feature.to(self.device)
        inclusion_vec = torch.mean(inclusion_feature, 0)
        inclusion_vec = inclusion_vec.view(1,-1)
        exclusion_vec = torch.mean(exclusion_feature, 0)
        exclusion_vec = exclusion_vec.view(1,-1)
        return inclusion_vec, exclusion_vec 

    def forward(self, in_ex_feature):
        result = [self.forward_single(in_mat, ex_mat) for in_mat, ex_mat in in_ex_feature]
        inclusion_mat = [in_vec for in_vec, ex_vec in result]
        inclusion_mat = torch.cat(inclusion_mat, 0)  #### 32,768
        exclusion_mat = [ex_vec for in_vec, ex_vec in result]
        exclusion_mat = torch.cat(exclusion_mat, 0)  #### 32,768 
        protocol_mat = torch.cat([inclusion_mat, exclusion_mat], 1)
        output = self.f(self.fc(protocol_mat))
        return output 

    @property
    def embedding_size(self):
        return self.output_dim

def vectorize(inputs, model):
    with torch.no_grad():
        outputs = model(**inputs)
    
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings

def save_sentence_bert_dict(device, filepath='data/bio_clinicalbert_embeddings.pkl', hf_model="emilyalsentzer/Bio_ClinicalBERT", limit = -1):
    protocols = get_all_protocols()
    sentences = set()
    for protocol in protocols:
        for part in split_protocol(protocol):
            sentences.update(part)
    
    protocol_sentence_2_embedding = {}
    
    if limit > 0:
        sentences = list(sentences)[:limit]
        
    tokenizer = AutoTokenizer.from_pretrained(hf_model, ignore_mismatched_sizes=True)
    model = AutoModel.from_pretrained(hf_model, ignore_mismatched_sizes=True)
    model.to(device)
        
    for sentence in tqdm(sentences, desc="Embedding sentences"):
        inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        embedding = vectorize(inputs, model)
        protocol_sentence_2_embedding[sentence] = embedding.cpu()
    
    with open(filepath, 'wb') as f:
        pickle.dump(protocol_sentence_2_embedding, f)

def load_sentence_2_vec(filepath='data/bio_clinicalbert_embeddings.pkl'):
    with open(filepath, 'rb') as f:
        return pickle.load(f)