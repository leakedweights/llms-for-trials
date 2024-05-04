import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from .layers import FeedForward

class ProtocolEmbedding(nn.Module):
    def __init__(self, output_dim: int, hidden_dim: int, num_layers: int, hf_model: str):
        super(ProtocolEmbedding, self).__init__()
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.embedding_size = output_dim
        self.tokenizer = AutoTokenizer.from_pretrained(hf_model)
        self.model = AutoModel.from_pretrained(hf_model)
        
        self.ffn = FeedForward(self.model.config.hidden_size, hidden_dim, output_dim, num_layers)
        
    def forward(self, tokens):
        with torch.no_grad():
            inputs = create_sliding_windows(tokens)
            original_length = len(tokens['input_ids'])
            results = self.model(**{k:v.to(device) for k,v in inputs.items()}).last_hidden_state
            aggregated_results = aggregate_embeddings(results, 32, original_length)
            
        x = self.ffn(aggregated_results)
        return x
    
    def create_sliding_windows(sefl, data, window_size=512, step_size=32):
        if len(data['input_ids']) < window_size:
            return {k:torch.tensor(v)[None] for k,v in data.items()}
        input_ids = data['input_ids']
        attention_mask = data['attention_mask']
        input_id_windows = []
        attention_mask_windows = []

        for start in range(0, len(input_ids), step_size):
            end = start + window_size
            window_ids = input_ids[start:end]

            pad_len = max(0, window_size - len(window_ids))
            window_ids_padded = window_ids + [0] * pad_len
            input_id_windows.append(window_ids_padded)

            window_mask = attention_mask[start:end]
            window_mask_padded = window_mask + [0] * pad_len
            attention_mask_windows.append(window_mask_padded)

        return {'input_ids': torch.tensor(input_id_windows), 'attention_mask' : torch.tensor(attention_mask_windows)}
    
    def aggregate_embeddings(model_outputs, step_size, original_length):

        num_windows, window_size, embedding_dim = model_outputs.shape
        sequence_length = (num_windows - 1) * step_size + window_size

        sum_embeddings = torch.zeros((sequence_length, embedding_dim), dtype=torch.float32, device = model_outputs.device)
        count_embeddings = torch.zeros((sequence_length, 1), dtype=torch.float32, device = model_outputs.device)

        start_position = 0
        for output in model_outputs:
            end_position = start_position + window_size
            sum_embeddings[start_position:end_position] += output[:min(window_size, sequence_length - start_position)]
            count_embeddings[start_position:end_position] += 1.0
            start_position += step_size

        averaged_embeddings = sum_embeddings / count_embeddings.clamp(min=1)

        return averaged_embeddings[:original_length]