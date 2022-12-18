import numpy
import torch
import os
from utils.sts_eval_utils import get_similarity_scores_tensors
from utils.result_utils import InferenceRowSTS
from subprocess import Popen, PIPE

class EvaluationSTS:
    def __init__(self, eval_file):
        self.eval_file = eval_file
    
    def run_eval_file(self, cmd):
        stdout = Popen(cmd, shell=True, stdout=PIPE).stdout
        out = stdout.read().decode('utf-8')
        return out

    def evaluate(self, gt_scores, pred_scores, pred_fname, gt_fname, run_dir):
        # write gt and pred to files
        gt_file, pred_file = gt_fname, pred_fname

        # create directory if doesnt exist
        if not os.path.exists(os.path.join('output',run_dir)):
            os.makedirs(os.path.join('output',run_dir), exist_ok=True)

        with open(os.path.join('output',run_dir, gt_file),'w') as f:
            [f.write(str(gt)+"\n") for gt in gt_scores]
        
        is_tensor = isinstance(pred_scores, numpy.ndarray) or isinstance(pred_scores, torch.Tensor)
        is_arr = isinstance(pred_scores, list)

        with open(os.path.join('output',run_dir, pred_file),'w') as f:
            if is_tensor:
                [f.write(str(pred.item())+"\n") for pred in pred_scores]
            elif is_arr:
                [f.write(str(pred)+"\n") for pred in pred_scores]
            else:
                raise Exception('pred datatype not supported')

        # evaluate using file
        cmd = f'perl {self.eval_file} ./output/{run_dir}/{gt_file} ./output/{run_dir}/{pred_file}'
        eval_result = self.run_eval_file(cmd)

        # parse output
        score = eval_result.strip().split(":")[-1]
        return score

def STSEvaluate(mod, ds, model_type, model_name, run_id):
    inf_rows = []
    # evaluating for each file type separately
    for f_type in ds.file_data:
        f_data = ds.file_data[f_type]
        s1, s2 = ds.get_sentence_pairs(data=f_data)
        gt = ds.get_gt_scores(data=f_data)
        # get encodings
        encoding_s1 = mod.get_encodings(s1)
        encoding_s2 = mod.get_encodings(s2)
        p_scores = get_similarity_scores_tensors(encoding_s1, encoding_s2)

        # evaluation
        eval_test = EvaluationSTS(eval_file='.\scripts\correlation-noconfidence.pl')
        gt_fname = f'gt_{f_type}_{model_name}_{model_type}.txt'
        pred_fname = f'pred_{f_type}_{model_name}_{model_type}.txt'
        corr_score = eval_test.evaluate(gt_scores=gt,
                                        pred_scores=p_scores,
                                        gt_fname=gt_fname,
                                        pred_fname=pred_fname,
                                        run_dir = run_id)

        # inf row
        inf_row = InferenceRowSTS().set_run_id(run_id).\
                                set_model_type(model_type).\
                                set_model_name(model_name).\
                                set_corr_score(corr_score).\
                                set_dataset_type(f_type)
        inf_rows.append(inf_row)
        
    return inf_rows