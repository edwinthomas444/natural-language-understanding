import torch
import pandas as pd

# computes cosine similarity between two Torch tensors
def get_similarity_scores_tensors(emb1, emb2):
    cos_sim = torch.nn.CosineSimilarity(dim=1)
    val = cos_sim(emb1, emb2)
    return val



# utility class for storing inference output types
class InferenceRow:
    def __init__(self):
        self.attr_names = ['RUN ID', 'MODEL TYPE', 'MODEL NAME', 'CORRELATION SCORE']
        self.data = {k:[] for k in self.attr_names}
    def set_run_id(self,val):
        self.run_id = val
        self.data['RUN ID'] = self.run_id
        return self
    def set_model_name(self, val):
        self.model_name = val
        self.data['MODEL TYPE'] = self.model_name
        return self
    def set_model_type(self, val):
        self.model_type = val
        self.data['MODEL NAME'] = self.model_type
        return self
    def set_corr_score(self, val):
        self.corr_score = val
        self.data['CORRELATION SCORE'] = self.corr_score
        return self
    
class InferenceData:
    def __init__(self):
        self.rows = []
    def add_row(self, row: InferenceRow):
        self.rows.append(row.data)
    def get_df(self):
        return pd.DataFrame(columns=list(self.rows[0].keys()),
                            data=self.rows)
    def save_df(self,out_path):
        df = self.get_df()
        df.to_csv(out_path)