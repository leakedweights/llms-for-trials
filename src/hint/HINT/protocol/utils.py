import csv
import torch

def get_all_protocols(filepath='data/raw_data.csv'):
    """Reads protocols from a CSV file, skipping the header."""
    with open(filepath, 'r') as csvfile:
        return [row[9] for row in csv.reader(csvfile)][1:]

def clean_protocol(protocol):
    """Cleans and splits the protocol text by lines, removing empty lines."""
    protocol = protocol.lower().split('\n')
    return [line.strip() for line in protocol if line.strip()]

def split_protocol(protocol):
    """Splits protocol text into inclusion and exclusion criteria."""
    protocol_split = clean_protocol(protocol)
    inclusion_idx = exclusion_idx = len(protocol_split)
    
    for idx, sentence in enumerate(protocol_split):
        if "inclusion" in sentence: inclusion_idx = idx
        if "exclusion" in sentence: exclusion_idx = idx 
    
    if inclusion_idx < exclusion_idx < len(protocol_split):
        return protocol_split[inclusion_idx:exclusion_idx], protocol_split[exclusion_idx:]
    return (protocol_split,)

def protocol2feature(protocol, sentence_2_vec):
    """Converts protocol text into feature vectors using precomputed embeddings."""
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





