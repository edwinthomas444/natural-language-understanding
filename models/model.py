from sentence_transformers import SentenceTransformer
import numpy as np
import gensim.downloader as api
from tqdm import tqdm 
from nltk.tokenize import word_tokenize
import torch
import gensim
import time
import nltk
import string
import torch
from torch import nn
from transformers import AutoModel, AutoConfig, AutoTokenizer
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch import nn
import torch.nn.functional as F

nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords

class Word2VecModel:
    def __init__(self, model_name, device):
        self.model_name = model_name
        self.device = device
        self.model = self.get_model()
        self.vocab = list(self.model.key_to_index.keys())

    def get_model(self):
        model = api.load(self.model_name)
        return model
    
    def get_word_tokens(self, sent):
        w_t = word_tokenize(sent)
        w_t_p = [w.lower() for w in w_t if w not in string.punctuation]
        # s_w = set(stopwords.words('english'))
        # w_t_ps = [w for w in w_t_p if not w.lower() in s_w]
        return w_t_p
    
    def get_word_embedding(self,token):
        return self.model[token]
    
    def get_filtered_toks(self,tok_list):
        filt_toks = [x for x in tok_list if x in self.vocab]
        return filt_toks
                
    def compute_sent_emb(self, word_emb):
        sent_emb = np.mean(word_emb, axis=0, keepdims=True)
        return sent_emb

    def get_encodings(self, data, tensor=True):
        all_emb = []
        for sent in tqdm(data,total=len(data)):
            # get word tokens
            w_t = self.get_word_tokens(sent)
            w_tf = self.get_filtered_toks(w_t)
            if len(w_tf)==0:
                w_tf = ['test']
            w_e = self.get_word_embedding(w_tf)
            s_e = self.compute_sent_emb(w_e)
            all_emb.append(s_e)
        all_emb = np.concatenate(all_emb, axis=0)
        if tensor:
            all_emb = torch.from_numpy(all_emb)
        return all_emb

class Doc2VecModel:
    def __init__(self, model_name, device):
        self.model = gensim.models.doc2vec.Doc2Vec(vector_size=300, min_count=2, epochs=200)

    def read_corpus(self, total, tokens_only=False):
        for i, line in enumerate(total):
            tokens = gensim.utils.simple_preprocess(line)
            if tokens_only:
                yield tokens
            else:
                # For training data, add tags
                yield gensim.models.doc2vec.TaggedDocument(tokens, [i])

    def train_model(self, s1,s2):
        total = []
        total.extend(s1)
        total.extend(s2)
        train_corpus = list(self.read_corpus(total))
        test_corpus = list(self.read_corpus(total, tokens_only=True)) 
        self.model.build_vocab(train_corpus)
        self.model.train(train_corpus, total_examples=self.model.corpus_count, epochs=self.model.epochs)


    def get_encodings(self, data, tensor=True):
        all_emb = []
        for sent in tqdm(data,total=len(data)):
            # get word tokens
            w_t = word_tokenize(sent)
            vec = self.model.infer_vector(w_t)
            all_emb.append(np.reshape(vec,(1,300)))
        all_emb = np.concatenate(all_emb, axis=0)
        if tensor:
            all_emb = torch.from_numpy(all_emb)
        return all_emb

class SentenceTransformerModel:
    def __init__(self, model_name, device='cuda'):
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


class QAModel(nn.Module):
    def __init__(self,
                 base_model_name,*args,**kwargs):
        
        super(QAModel,self).__init__()
        self.base_model = AutoModel.from_pretrained(base_model_name,add_pooling_layer=False)
        self.base_model_config = AutoConfig.from_pretrained(base_model_name)
        # define QA head (2 dim, one for start and other for end span)
        self.qa_head = nn.Linear(self.base_model_config.hidden_size, 2)
        
    def forward(self,
                input_ids,
                attention_mask,
                token_type_ids,
                start_positions=None,
                end_positions=None):
        
        out = self.base_model(input_ids,
                             attention_mask=None,
                             token_type_ids=None)
        
        logits = self.qa_head(out[0])
        
        start_logits, end_logits = torch.split(logits, split_size_or_sections=1, dim=-1)
        start_logits, end_logits = start_logits.squeeze(dim=-1).contiguous(), end_logits.squeeze(dim=-1).contiguous()
        
        total_loss = None
        if start_positions is not None and end_positions is not None:
            # compute loss
            ignore_index = start_logits.size(1)
            start_positions = start_positions.clamp(0,ignore_index)
            end_positions = end_positions.clamp(0,ignore_index)
            loss_fct = CrossEntropyLoss(ignore_index=ignore_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
        
        out_dict = {
            'loss':total_loss,
            'start_logits':start_logits,
            'end_logits':end_logits,
            'hidden_states':out.hidden_states,
            'attention':out.attentions,
        }
        
        return out_dict


class MetaLearner(nn.Module):
    def __init__(self, n_best, num_base_models, feat_dim):
        super(MetaLearner,self).__init__()
        self.hidden_layer_size = int(num_base_models*n_best*feat_dim/2.0)
        self.dense1 = nn.Linear(num_base_models*n_best*feat_dim, self.hidden_layer_size)
        self.dense2 = nn.Linear(self.hidden_layer_size, num_base_models*n_best)
    def forward(self, data):
        # data is tensor of size (b, 2*n_best, 2 features)
        flattened = torch.flatten(data, start_dim=1) # flatten after batch (b, n_best*2)
        out=self.dense1(flattened)
        # adding non-linearity
        out=F.relu(out)
        out=self.dense2(out)
        out=F.log_softmax(out, dim=1)
        return out
        