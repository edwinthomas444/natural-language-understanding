import sys
sys.path.append('./')

from tqdm import tqdm
import os
import json
from dataset.dataset import DatasetSQUAD
from preprocess.squad_processors import SquadProcessor
from transformers import AutoTokenizer
from preprocess.squad_features import squad_convert_examples_to_features
from torch.utils.data import DataLoader, SequentialSampler
import torch 
from preprocess.squad_objects import *
import numpy as np
from utils.squad_eval_utils import compute_predictions_logits, squad_evaluate
from collections import OrderedDict
import copy
import torch
from torch import nn
from models.model import MetaLearner
import torch.nn.functional as F
from models.model import QAModel


def get_meta_features(mod, batch, features, examples, n_best_val, tokenizer, batch_count, lower_case, eval):
    all_results = []
    with torch.no_grad():
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "token_type_ids": batch[2],
        }
        feature_indices = batch[3]
        # print('feature_indices: ',feature_indices)
        outputs = mod(**inputs)

        for i, feature_index in enumerate(feature_indices):
            eval_feature = features[feature_index.item()]
            unique_id = int(eval_feature.unique_id)
            start_logits = outputs['start_logits'][i].tolist()
            end_logits = outputs['end_logits'][i].tolist()
            result = SquadResult(unique_id, start_logits, end_logits)
            # print(result.start_logits, result.end_logits, result.unique_id)
            all_results.append(result)
    
    pred_file = os.path.join(os.getcwd(),'output',"ensemble_test_pred.json")
    nbest_file = os.path.join(os.getcwd(),'output',"ensemble_nbest_pred.json")
    null_log_file = os.path.join(os.getcwd(),'output',"ensemble_nbest_nulllog.json")
    # print(all_results[0].start_logits, all_results[0].end_logits)
    # exit(0)
    ## logic for post_processing as input to ensemble
    predictions, all_nbest_json, example_subset = compute_predictions_logits(
                                                        examples,
                                                        features,
                                                        all_results,
                                                        n_best_val, # nbest
                                                        30, # max answer length
                                                        lower_case, # do lower case
                                                        pred_file,
                                                        nbest_file,
                                                        null_log_file,
                                                        False, # verbose logging
                                                        True, # version with negatives
                                                        0, tokenizer)
    
                     
    # print('\nBefore: all_nbest_json',all_nbest_json)
    ##### for creating dummies when nbest less than 8
    dummy = OrderedDict([('probability', 0.0), 
                        ('start_logit', 0.0), 
                        ('end_logit', 0.0),
                        ('text', 'empty')])
    
    for qas_id in all_nbest_json:
        total_nbest_vals = len(all_nbest_json[qas_id])
        for i in range(0,n_best_val-total_nbest_vals):
            all_nbest_json[qas_id].append(copy.deepcopy(dummy))


    nbest = {i:OrderedDict() for i in range(0,n_best_val,1)}
    for key in all_nbest_json:
        # ind loops through n_best indexes that are populated
        for ind, item in enumerate(all_nbest_json[key]):
            if ind<n_best_val:
                nbest[ind][key]=item['text']

    if not eval:
        results = {key:OrderedDict() for key in nbest.keys()}

        for key in tqdm(nbest, total=len(list(nbest.keys()))):
            for example in example_subset:
                # get the qas id
                qas_id = example.qas_id
                od = OrderedDict()
                od[qas_id]=nbest[key][qas_id] # text
                results[key][qas_id]=squad_evaluate([example], od)['f1']
                
        # merge with n_best_all
        for qas_id in all_nbest_json:
            for n_best_ind, n_best_pred in enumerate(all_nbest_json[qas_id]):
                # get pred f1 scores for those qas_ids
                f1_score = results[n_best_ind][qas_id]
                all_nbest_json[qas_id][n_best_ind]['f1']=f1_score

        # write to file
        if batch_count%50 == 0:
            with open(os.path.join(os.getcwd(),'output',f"ensemble_test_predCombined_{batch_count}.json"), "w") as writer:
                writer.write(json.dumps(all_nbest_json, indent=4) + "\n")
        
    # create tensors with features for a batch
    # (b, n_best, 2) || (b, n_best, 2) -> (b, 2*n_best, 2)  -> meta model-> (1 x m*n)
    # gt for (b, n_best, 2) -> (b, n_best)
    
    # for first model get (b, n_best, 2)
    batch_features = []
    batch_gt = []
    for batch_elem_key in all_nbest_json:
        batch_elem = all_nbest_json[batch_elem_key]
        n_b_features = []
        n_b_gt = []
        for n_b in batch_elem:
            prob = n_b['probability']
            # normalize by maximum answer length
            text_len = float(len(n_b['text'].split())/30.0)
            start_logit = n_b['start_logit']
            end_logit = n_b['end_logit']
            
            # feature_nb = np.expand_dims(np.array([prob, text_len]),axis=0)
            feature_nb = np.expand_dims(np.array([prob, start_logit, end_logit, text_len]),axis=0)
            n_b_features = np.concatenate((n_b_features, feature_nb), axis=0) if isinstance(n_b_features, np.ndarray) else feature_nb

            if not eval:
                f1 = n_b['f1']
                n_b_gt.append(f1)
                
        n_b_features = np.expand_dims(n_b_features, axis=0)
        batch_features = np.concatenate((batch_features, n_b_features), axis=0) if isinstance(batch_features, np.ndarray) else n_b_features
            
        # for gt
        if not eval:
            n_b_gt = np.expand_dims(np.array(n_b_gt), axis=0)
            batch_gt = np.concatenate((batch_gt, n_b_gt), axis=0) if isinstance(batch_gt, np.ndarray) else n_b_gt

    # b*n_best*2,   b*n_best
    if not eval:
        return [batch_features, batch_gt]
    else:
        return [batch_features, all_nbest_json]

def ensembleTrain():
    # get from config
    n_best_val = 2
    feat_dim = 4

    ds = DatasetSQUAD(download_url=None)
    processor = SquadProcessor(train_file=ds.train_file, dev_file=ds.test_file)
    
    examples = processor.get_dev_examples(ds.dataset_root,'train-v2.0.json')
    # reducing to 20 % of train set for training the meta model
    sub_sample = int(len(examples)*0.2)
    examples = examples[:sub_sample]

    # get subset of examples
    device = torch.device('cuda')

    # bert base trained 1 epochs
    tokenizer_b0 = AutoTokenizer.from_pretrained('roberta-base',use_fast=False)
    mod_b0 = QAModel(base_model_name='roberta-base')
    mod_b0.load_state_dict(torch.load('../../NLP-Project/QAModel_roberta-base.pt'))
    mod_b0.to(device)
    mod_b0.eval()
    # bert base trained 2 epochs #albert-base-v2 ../../NLP-Project/QAModel_albert-base-v2.pt
    tokenizer_b1 = AutoTokenizer.from_pretrained('roberta-base',use_fast=False)
    mod_b1 = QAModel(base_model_name='roberta-base')
    mod_b1.load_state_dict(torch.load('./output/QA_roberta_full\QAModel_roberta-base.pt'))
    mod_b1.to(device)
    mod_b1.eval()

    features_b0, dataset_b0 = squad_convert_examples_to_features(
                                                                examples=examples,
                                                                tokenizer=tokenizer_b0,
                                                                max_seq_length=384,
                                                                doc_stride=128,
                                                                max_query_length=64,
                                                                is_training=False
                                                            )
    # samplers
    eval_sampler_b0 = SequentialSampler(dataset_b0)
    eval_dataloader_b0 = DataLoader(dataset_b0, sampler=eval_sampler_b0, batch_size=2)

    features_b1, dataset_b1 = squad_convert_examples_to_features(
                                                                examples=examples,
                                                                tokenizer=tokenizer_b1,
                                                                max_seq_length=384,
                                                                doc_stride=128,
                                                                max_query_length=64,
                                                                is_training=False
                                                            )
    # samplers
    eval_sampler_b1 = SequentialSampler(dataset_b1)
    eval_dataloader_b1 = DataLoader(dataset_b1, sampler=eval_sampler_b1, batch_size=2)

    

    kl_loss = nn.KLDivLoss(reduction="batchmean")

    # load the meta learner model
    meta_learner_mod = MetaLearner(n_best=n_best_val, num_base_models=2, feat_dim=feat_dim)
    meta_learner_mod.to(device)
    # print(meta_learner_mod)

    count = 0
    total_epochs = 1
    for epoch in range(total_epochs):
        print(f'\n Epoch {epoch}')
        for batch_b0, batch_b1 in tqdm(zip(eval_dataloader_b0, eval_dataloader_b1), desc='Training Meta Learner'):

            count+=1
            # layer-0 models
            mod_b0.eval()
            mod_b1.eval()

            batch_b0 = tuple(t.to(device) for t in batch_b0)
            batch_b1 = tuple(t.to(device) for t in batch_b1)

            # b*n_best*2,   b*n_best
            X_b0, y_b0 = get_meta_features(mod_b0, batch_b0, features_b0, examples, n_best_val, tokenizer_b0, batch_count=count, lower_case = False, eval=False)
            X_b1, y_b1 = get_meta_features(mod_b1, batch_b1, features_b1, examples, n_best_val, tokenizer_b1, batch_count=count, lower_case = True, eval=False)

            try:
                inp_meta = torch.from_numpy(np.concatenate((X_b0, X_b1), axis=1)).float()
                target_meta = torch.from_numpy(np.concatenate((y_b0, y_b1), axis=1)).float()
                target_meta = F.softmax(target_meta, dim=1)
            except Exception:
                print('x_b0 shape: ',X_b0.shape, 'x_b1 shape: ', X_b1.shape)
                print('y_b0 shape: ',y_b0.shape, 'y_b1 shape: ', y_b1.shape)
                continue

            # move meta train batch to gpu
            inp_meta = inp_meta.to(device)
            target_meta = target_meta.to(device)

            # print(inp_meta)
            meta_out = meta_learner_mod(inp_meta)
            # print(out.shape, out)
            # print(meta_out.shape, target_meta.shape)
            loss_out = kl_loss(meta_out, target_meta)

            loss_out.backward()
            if count%15 == 0:
                print('\nTotal loss: ',loss_out.item())
            meta_learner_mod.zero_grad()
        print('\nSaving model')
        save_path = f'../../NLP-Project/ensemble_train1_e{epoch}.pt'
        torch.save(meta_learner_mod.state_dict(),f'../../NLP-Project/ensemble_train1_e{epoch}.pt')
    return save_path

def ensembleEvaluate(ensemble_path):
    # get from config
    n_best_val = 2
    feat_dim = 4

    ds = DatasetSQUAD(download_url=None)
    processor = SquadProcessor(train_file=ds.train_file, dev_file=ds.test_file)
    
    examples = processor.get_dev_examples(ds.dataset_root,'dev-v2.0.json')
    # reducing to 20 % of train set for training the meta model
    sub_sample = int(len(examples)*1.0)
    examples = examples[:sub_sample]

    device = torch.device('cuda')

    # bert base trained 1 epochs
    tokenizer_b0 = AutoTokenizer.from_pretrained('roberta-base',use_fast=False)
    mod_b0 = QAModel(base_model_name='roberta-base')
    mod_b0.load_state_dict(torch.load('../../NLP-Project/QAModel_roberta-base.pt'))
    mod_b0.to(device)
    mod_b0.eval()
    # bert base trained 2 epochs
    # bert base trained 2 epochs #albert-base-v2 ../../NLP-Project/QAModel_albert-base-v2.pt
    tokenizer_b1 = AutoTokenizer.from_pretrained('roberta-base',use_fast=False)
    mod_b1 = QAModel(base_model_name='roberta-base')
    mod_b1.load_state_dict(torch.load('./output/QA_roberta_full\QAModel_roberta-base.pt'))
    mod_b1.to(device)
    mod_b1.eval()

    features_b0, dataset_b0 = squad_convert_examples_to_features(
                                                                examples=examples,
                                                                tokenizer=tokenizer_b0,
                                                                max_seq_length=384,
                                                                doc_stride=128,
                                                                max_query_length=64,
                                                                is_training=False
                                                            )
    # samplers
    eval_sampler_b0 = SequentialSampler(dataset_b0)
    eval_dataloader_b0 = DataLoader(dataset_b0, sampler=eval_sampler_b0, batch_size=2)

    features_b1, dataset_b1 = squad_convert_examples_to_features(
                                                                examples=examples,
                                                                tokenizer=tokenizer_b1,
                                                                max_seq_length=384,
                                                                doc_stride=128,
                                                                max_query_length=64,
                                                                is_training=False
                                                            )
    # samplers
    eval_sampler_b1 = SequentialSampler(dataset_b1)
    eval_dataloader_b1 = DataLoader(dataset_b1, sampler=eval_sampler_b1, batch_size=2)

    # load the meta learner model
    meta_learner_mod = MetaLearner(n_best=n_best_val, num_base_models=2, feat_dim=feat_dim)
    meta_learner_mod.load_state_dict(torch.load(ensemble_path))
    meta_learner_mod.to(device)
    meta_learner_mod.eval()

    count = 0
    all_nbest_selected = OrderedDict()
    for batch_b0, batch_b1 in tqdm(zip(eval_dataloader_b0, eval_dataloader_b1), desc='Evaluating Meta Learner'):
        count+=1
        # layer-0 models
        mod_b0.eval()
        mod_b1.eval()

        batch_b0 = tuple(t.to(device) for t in batch_b0)
        batch_b1 = tuple(t.to(device) for t in batch_b1)

        # print("examples length before: ",len(examples))

        # b*n_best*2,   b*n_best
        X_b0, all_nbest_json_b0 = get_meta_features(mod_b0, batch_b0, features_b0, examples, n_best_val, tokenizer_b0, batch_count=count, lower_case=True, eval=True)
        X_b1, all_nbest_json_b1 = get_meta_features(mod_b1, batch_b1, features_b1, examples, n_best_val, tokenizer_b1, batch_count=count, lower_case=True, eval=True)

        
        with torch.no_grad():
            try:
                inp_meta = torch.from_numpy(np.concatenate((X_b0, X_b1), axis=1)).float()
            except Exception:
                print('x_b0 shape: ',X_b0.shape, 'x_b1 shape: ', X_b1.shape)
                continue
            # inp_meta = torch.from_numpy(np.concatenate((X_b0, X_b1), axis=1)).float()

            # move meta train batch to gpu
            inp_meta = inp_meta.to(device)

            # print(inp_meta.shape)
            meta_out = meta_learner_mod(inp_meta)

            sorted, indices = torch.sort(meta_out, descending=True)
            models_order = torch.split(indices, split_size_or_sections=1, dim=0)

            for bat_num, bat_indices in enumerate(models_order):
                np_bat_indices = bat_indices.cpu().detach().numpy()[0]
                top_nbest = np_bat_indices[:n_best_val]
                top_nbest = [0,1,2,3]
                bat_qas_id = list(all_nbest_json_b0.items())[bat_num][0]
                m0_nbest = list(all_nbest_json_b0.items())[bat_num][1]
                m1_nbest = list(all_nbest_json_b1.items())[bat_num][1]
                combined = m0_nbest + m1_nbest
                combined_nbest = [combined[int(ind)] for ind in top_nbest]
                all_nbest_selected[bat_qas_id] = combined_nbest
            
            # create an all n_best json from all base-0 models (by sorting on argmax)

    with open(os.path.join(os.getcwd(),'output',f"ensemble_selected_nbest.json"), "w") as writer:
        writer.write(json.dumps(all_nbest_selected, indent=4) + "\n")

    # taking top-1 prediction for prediction
    

    pred_selected = OrderedDict()
    for qas_id in all_nbest_selected:
        pred_selected[qas_id] = all_nbest_selected[qas_id][0]['text']

    # save pred_selected as best predictions of the ensemble
    with open(os.path.join(os.getcwd(),'output',f"ensemble_best_preds.json"), "w") as writer:
        writer.write(json.dumps(pred_selected, indent=4) + "\n")
    
    # evaluate and get results
    results = squad_evaluate(examples, pred_selected)
    print("\nResults: ",results)
            

def main():
    # training the ensemble
    # save_path = ensembleTrain()

    save_path = '../../NLP-Project/ensemble_train1_e0.pt'
    # evaluate the ensemble
    ensembleEvaluate(ensemble_path = save_path)
    

if __name__ == '__main__':
    main()