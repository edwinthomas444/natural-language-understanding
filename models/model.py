from sentence_transformers import SentenceTransformer
import numpy as np
import gensim.downloader as api
from tqdm import tqdm 
from nltk.tokenize import word_tokenize
import torch
import time

class Word2VecModel:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = self.get_model()
        self.vocab = list(self.model.key_to_index.keys())

    def get_model(self):
        model = api.load(self.model_name)
        return model
    
    def get_word_tokens(self, sent):
        w_t = word_tokenize(sent)
        return w_t
    
    def get_word_embedding(self,token):
        return self.model[token]
    
    def get_filtered_toks(self,tok_list):
        filt_toks = [x for x in tok_list if x in self.vocab]
        return filt_toks
                
    def compute_sent_emb(self, word_emb):
        sent_emb = np.average(word_emb, axis=0, keepdims=True)
        return sent_emb

    def get_encodings(self, data, tensor=True):
        all_emb = []
        for sent in tqdm(data,total=len(data)):
            # get word tokens
            w_t = self.get_word_tokens(sent)
            w_tf = self.get_filtered_toks(w_t)
            w_e = self.get_word_embedding(w_tf)
            s_e = self.compute_sent_emb(w_e)
            all_emb.append(s_e)
        all_emb = np.concatenate(all_emb, axis=0)
        if tensor:
            all_emb = torch.from_numpy(all_emb)
        return all_emb

class SentenceTransformerModel:
    def __init__(self, model_name, device='cpu'):
        self.model_name = model_name
        self.device = device
        self.model = self.get_transformer()
        
    def get_transformer(self):
        model = SentenceTransformer(self.model_name, device=self.device)
        return model
    
    def get_batched_data(self, iterable, bs=1):
        len_iter = len(iterable)
        for idx in range(0, len_iter, bs):
            yield iterable[idx:min(idx + bs, len_iter)]

    def get_encodings(self, data, bs = 64):
        # computes bs embeddings and returns stacked output
        model_out = self.model.encode(data,
                                     batch_size=bs,
                                     show_progress_bar=True,
                                     device=self.device,
                                     convert_to_tensor=True)
        return model_out