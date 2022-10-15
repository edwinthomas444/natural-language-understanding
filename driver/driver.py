import sys
sys.path.append('./')

from dataset.dataset import DatasetSTS
from models.model import SentenceTransformerModel, Word2VecModel
from utils.evaluation import EvaluationSTS
from utils.utility import get_similarity_scores_tensors, InferenceRow, InferenceData
from models.model_maps import get_model
import json
import itertools
import datetime


def config_parser(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def driver():
    # pre-process data
    ds = DatasetSTS(download_url='http://alt.qcri.org/semeval2016/task1/data/uploads/sts2016-english-with-gs-v1.0.zip')

    ds.pre_process()
    s1, s2 = ds.get_sentence_pairs()
    gt = ds.get_gt_scores()

    # get config for runs
    config_path = './configs/run_config.json' # TODO: creating argparser
    all_configs = config_parser(config_path=config_path)

    # create Infout obj
    inf_out = InferenceData()
    for run in all_configs:
        run_id = run['run_name']
        eval_dict = run['eval']
        for m_dict in eval_dict:
            m_type = m_dict['model_type']
            m_names = m_dict['model_names']
            all_combs = itertools.product([m_type], m_names)
            
            for (m_t, m_n) in all_combs:
                mod = get_model(m_t, m_n)
                # get encodings
                encoding_s1 = mod.get_encodings(s1)
                encoding_s2 = mod.get_encodings(s2)
                p_scores = get_similarity_scores_tensors(encoding_s1, encoding_s2)

                # evaluation
                eval_test = EvaluationSTS(eval_file='.\scripts\correlation-noconfidence.pl')
                corr_score = eval_test.evaluate(gt_scores=gt, pred_scores=p_scores)

                # inf row
                inf_row = InferenceRow().set_run_id(run_id).\
                                        set_model_type(m_t).\
                                        set_model_name(m_n).\
                                        set_corr_score(corr_score)
                inf_out.add_row(inf_row)
    
    # save to csv
    out_file_stamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    inf_out.save_df(out_path=f'./output/{out_file_stamp}.csv')

def main():
    # ToDo: Argument Parser defn. 
    driver()

if __name__ == '__main__':
    main()