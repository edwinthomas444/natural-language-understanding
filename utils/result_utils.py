import torch
import pandas as pd
from enum import Enum


# utility class for storing inference output types
class InferenceRowSTS:
    def __init__(self):
        self.attr_names = ['RUN ID', 'DATASET TYPE', 
                           'MODEL TYPE', 'MODEL NAME', 
                           'CORRELATION SCORE']
        self.data = {k:[] for k in self.attr_names}
    def set_run_id(self,val):
        self.run_id = val
        self.data['RUN ID'] = self.run_id
        return self
    def set_model_name(self, val):
        self.model_name = val
        self.data['MODEL NAME'] = self.model_name
        return self
    def set_model_type(self, val):
        self.model_type = val
        self.data['MODEL TYPE'] = self.model_type
        return self
    def set_corr_score(self, val):
        self.corr_score = val
        self.data['CORRELATION SCORE'] = self.corr_score
        return self
    def set_dataset_type(self, val):
        self.dataset_type = val
        self.data['DATASET TYPE'] = self.dataset_type
        return self
    
class InferenceRowSQUAD:
    def __init__(self):
        self.attr_names = ['RUN ID', 'MODEL TYPE', 'MODEL NAME', 
                           'EM','F1','HasAns_EM','HasAns_F1',
                           'NoAns_EM','NoAns_F1','Best_EM',
                           'Best_EM_thresh','Best_F1','Best_F1_thresh']
        self.data = {k:[] for k in self.attr_names}
    def set_run_id(self,val):
        self.run_id = val
        self.data['RUN ID'] = self.run_id
        return self
    def set_model_name(self, val):
        self.model_name = val
        self.data['MODEL NAME'] = self.model_name
        return self
    def set_model_type(self, val):
        self.model_type = val
        self.data['MODEL TYPE'] = self.model_type
        return self
    def set_em_type(self, val):
        self.em_type = val
        self.data['EM'] = self.em_type
        return self
    def set_f1_type(self, val):
        self.f1_type = val
        self.data['F1'] = self.f1_type
        return self
    def set_hasans_em_type(self, val):
        self.hasans_em_type = val
        self.data['HasAns_EM'] = self.hasans_em_type
        return self
    def set_hasans_f1_type(self, val):
        self.hasans_f1_type = val
        self.data['HasAns_F1'] = self.hasans_f1_type
        return self
    def set_noans_em_type(self, val):
        self.noans_em_type = val
        self.data['NoAns_EM'] = self.noans_em_type
        return self
    def set_noans_f1_type(self, val):
        self.noans_f1_type = val
        self.data['NoAns_F1'] = self.noans_f1_type
        return self
    def set_best_em_type(self, val):
        self.best_em_type = val
        self.data['Best_EM'] = self.best_em_type
        return self
    def set_best_emthresh_type(self, val):
        self.best_emthresh_type = val
        self.data['Best_EM_thresh'] = self.best_emthresh_type
        return self
    def set_best_f1_type(self, val):
        self.best_f1_type = val
        self.data['Best_F1'] = self.best_f1_type
        return self
    def set_best_f1thresh_type(self, val):
        self.best_f1thresh_type = val
        self.data['Best_F1_thresh'] = self.best_f1thresh_type
        return self
    
    
       

class InferenceData:
    def __init__(self):
        self.rows = []
    def add_rows(self, rows):
        if isinstance(rows, list):
            [self.rows.append(row.data) for row in rows]
        else:
            self.rows.append(rows.data)
    def get_df(self):
        return pd.DataFrame(columns=list(self.rows[0].keys()),
                            data=self.rows)
    def save_df(self,out_path):
        df = self.get_df()
        df.to_csv(out_path)