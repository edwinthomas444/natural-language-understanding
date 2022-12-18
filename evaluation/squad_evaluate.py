from utils.squad_eval_utils import *
from utils.result_utils import InferenceRowSQUAD
from preprocess.squad_features import *
from preprocess.squad_objects import *
from preprocess.squad_processors import *
from preprocess.squad_utilities import *
from transformers import AutoModel, AutoConfig, AutoTokenizer
from torch.utils.data import DataLoader, SequentialSampler

def QAEvaluate(mod, f_params, e_params, ds, model_type, model_name, run_id, device='cpu'):
    device = torch.device(device)

    cache_file = 'test_features_{}_{}_{}'.format(f_params['max_query_length'],
                                                 f_params['doc_stride'],
                                                 f_params['max_query_length'])
    if f_params['load_from_cache_test']:
        print('\nLoading test features from cache..\n')
        if os.path.isfile(os.path.join(os.getcwd(),'cache',cache_file)):
            cached_tensor = torch.load(os.path.join(os.getcwd(),'cache',cache_file))
            features, dataset, examples = cached_tensor['features'], cached_tensor['dataset'], cached_tensor['examples']
            sub_sample = int(len(examples)*e_params['subset_samples'])
            examples = examples[:sub_sample]
        else:
            raise Exception('Cache File doesnt exist..')
    else:
        # get test examples
        processor = SquadProcessor(train_file=ds.train_file, dev_file=ds.test_file)
        examples = processor.get_dev_examples(ds.dataset_root,'dev-v2.0.json')
        sub_sample = int(len(examples)*e_params['subset_samples'])
        examples = examples[:sub_sample]

        # create features
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        features, dataset = squad_convert_examples_to_features(
                            examples=examples,
                            tokenizer=tokenizer,
                            max_seq_length=f_params['max_seq_length'],
                            doc_stride=f_params['doc_stride'],
                            max_query_length=f_params['max_query_length'],
                            is_training=False
                        )

        # cache the data
        print('\nCaching Test Features..\n')
        if not os.path.exists(os.path.join(os.getcwd(),'cache')):
            os.makedirs(os.path.join(os.getcwd(),'cache'), exist_ok=True)

        torch.save({"features": features, "dataset": dataset, "examples":examples}, 
                   os.path.join(os.getcwd(),'cache',cache_file))

    # defining loaders
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=e_params['batch_size'])

    # evaluate (load from path if string specified, else use model object)
    if isinstance(mod, str):
        mod = torch.load(mod)
    
    print('\n Evaluating Model.... \n')
    all_results = []
    for batch in tqdm(eval_dataloader, desc='Evaluating model'):
        mod.eval()
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }
            feature_indices = batch[3]
            outputs = mod(**inputs)

            for i, feature_index in enumerate(feature_indices):
                eval_feature = features[feature_index.item()]
                unique_id = int(eval_feature.unique_id)
                
                start_logits = outputs['start_logits'][i].tolist()
                end_logits = outputs['end_logits'][i].tolist()

                result = SquadResult(unique_id, start_logits, end_logits)
                all_results.append(result)
    
    # Compute predictions
    output_prediction_file = f"predictions_{model_name}_{model_type}.json"
    output_nbest_file = f"nbest_predictions_{model_name}_{model_type}.json"
    output_null_log_odds_file = f"null_log_{model_name}_{model_type}.json"

    if not os.path.exists(os.path.join(os.getcwd(),'output',run_id)):
        os.makedirs(os.path.join(os.getcwd(),'output',run_id), exist_ok=True)

    pred_file = os.path.join(os.getcwd(),'output',run_id, output_prediction_file)
    nbest_file = os.path.join(os.getcwd(),'output',run_id, output_nbest_file)
    null_log_file = os.path.join(os.getcwd(),'output',run_id, output_null_log_odds_file)

    predictions = compute_predictions_logits(
                    examples,
                    features,
                    all_results,
                    e_params['n_best'], # nbest
                    e_params['max_answer_length'], # max answer length
                    e_params['do_lower_case'], # do lower case
                    pred_file,
                    nbest_file,
                    null_log_file,
                    False, # verbose logging
                    True, # version with negatives
                    0, tokenizer)

    results = squad_evaluate(examples, predictions)
    results = list(results.items())
    r_dict = {}
    for k,v in results:
        r_dict[k] = v
    

    inf_row = InferenceRowSQUAD().set_run_id(run_id).\
                                set_model_type(model_type).\
                                set_model_name(model_name).\
                                set_f1_type(r_dict['f1']).\
                                set_em_type(r_dict['exact']).\
                                set_hasans_em_type(r_dict['HasAns_exact']).\
                                set_hasans_f1_type(r_dict['HasAns_f1']).\
                                set_noans_em_type(r_dict['NoAns_exact']).\
                                set_noans_f1_type(r_dict['NoAns_f1']).\
                                set_best_em_type(r_dict['best_exact']).\
                                set_best_emthresh_type(r_dict['best_exact_thresh']).\
                                set_best_f1_type(r_dict['best_f1']).\
                                set_best_f1thresh_type(r_dict['best_f1_thresh'])
                                
    return inf_row