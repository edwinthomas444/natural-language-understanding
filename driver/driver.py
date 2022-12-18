import sys
sys.path.append('./')

from dataset.dataset import DatasetSTS, DatasetSQUAD
from utils.result_utils import InferenceData
from models.model_maps import get_model

from train.squad_train import QATrain
from evaluation.squad_evaluate import QAEvaluate

from evaluation.sts_evaluate import STSEvaluate
import json
import itertools
import datetime


def config_parser(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def driver():
    # get config for runs
    config_path = './configs/run_config.json' # TODO: creating argparser
    all_configs = config_parser(config_path=config_path)

    for run in all_configs:
        # create Infout obj
        inf_out = InferenceData()

        task_type = run['task_type']
        run_id = run['run_name']
        
        train_dict, eval_dict = {}, {}
        if 'eval' in run:
            eval_dict = run['eval']
        if 'train' in run:
            train_dict = run['train']
        device = run['device']
        dataset_configs = run['dataset']

        download_url = None
        if run['dataset']['download']:
            download_url = dataset_configs['url']

        # Dataset object based on task-type
        if task_type == 'SentenceSimilarity':
            ds = DatasetSTS(download_url=download_url)
        elif task_type == 'QuestionAnswering':
            ds = DatasetSQUAD(download_url=download_url)
        else:
            raise Exception('Task Type not supported for Datasets')
        
        # for train
        if len(train_dict):
            m_type = train_dict['model_type']
            m_names = train_dict['model_names']
            f_params = train_dict['feature_params']
            t_params = train_dict['train_params']
            e_params = train_dict['eval_params']
            all_combs = itertools.product([m_type], m_names)
            for (m_t, m_n) in all_combs:
                mod = get_model(m_t, m_n, device=device)
                if task_type == 'QuestionAnswering':
                    inf_rows = QATrain(mod, f_params, t_params, e_params, ds, m_t, m_n, run_id, device)
                else:
                    raise Exception('Task type not supported for Training')
                inf_out.add_rows(inf_rows)

        # for evaluation
        for m_dict in eval_dict:
            m_type = m_dict['model_type']
            m_names = m_dict['model_names']
            all_combs = itertools.product([m_type], m_names)
            for (m_t, m_n) in all_combs:
                mod = get_model(m_t, m_n, device=device)
                # Dataset object based on task-type
                if task_type == 'SentenceSimilarity':
                    inf_rows = STSEvaluate(mod, ds, m_t, m_n, run_id)
                elif task_type == 'QuestionAnswering':
                    inf_rows = QAEvaluate(mod, ds, m_t, m_n, run_id)
                else:
                    raise Exception('Task Type not supported for Evaluation')
                
                inf_out.add_rows(inf_rows)
            
        # save to csv
        out_file_stamp = datetime.datetime.now().strftime('Results-%Y%m%d-%H%M%S')
        inf_out.save_df(out_path=f'./output/{run_id}/{out_file_stamp}.csv')

def main():
    # ToDo: Argument Parser defn. 
    driver()

if __name__ == '__main__':
    main()