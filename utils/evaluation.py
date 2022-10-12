import numpy
import torch
from subprocess import Popen, PIPE
import os
import subprocess


class EvaluationSTS:
    def __init__(self, eval_file):
        self.eval_file = eval_file
    
    def run_eval_file(self, cmd):
        stdout = Popen(cmd, shell=True, stdout=PIPE).stdout
        out = stdout.read().decode('utf-8')
        return out

    def evaluate(self, gt_scores, pred_scores):
        # write gt and pred to files
        gt_file, pred_file = 'gt.txt', 'pred.txt'

        with open(os.path.join('output','gt.txt'),'w') as f:
            [f.write(str(gt)+"\n") for gt in gt_scores]
        
        is_tensor = isinstance(pred_scores, numpy.ndarray) or isinstance(pred_scores, torch.Tensor)
        is_arr = isinstance(pred_scores, list)

        with open(os.path.join('output','pred.txt'),'w') as f:
            if is_tensor:
                [f.write(str(pred.item())+"\n") for pred in pred_scores]
            elif is_arr:
                [f.write(str(pred)+"\n") for pred in pred_scores]
            else:
                raise Exception('pred datatype not supported')

        # evaluate using file
        cmd = f'perl {self.eval_file} ./output/{gt_file} ./output/{pred_file}'
        eval_result = self.run_eval_file(cmd)

        # parse output
        score = eval_result.strip().split(":")[-1]
        return score