import torch
from transformers import AutoTokenizer, AutoModel
import pickle
from tqdm import tqdm
from .utils import get_all_protocols, split_protocol

def vectorize(inputs, model):
    with torch.no_grad():
        outputs = model(**inputs)
    
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings

def save_sentence_bert_dict(device, filepath='data/bio_clinicalbert_embeddings.pkl', hf_model="emilyalsentzer/Bio_ClinicalBERT", limit = -1):
    """Saves sentence embeddings to a pickle file."""
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
    """Loads sentence embeddings from a pickle file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)





