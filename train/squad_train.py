import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer
from torch.optim import SGD
from torch.utils.data import RandomSampler, DataLoader, Subset

from preprocess.squad_processors import SquadProcessor
from preprocess.squad_features import squad_convert_examples_to_features
from evaluation.squad_evaluate import QAEvaluate
import os

def QATrain(mod, f_params, t_params, e_params, ds, model_type, model_name, run_id, device=None):
    # ToDo: Logger Implementation
    print('\nTraining Model...\n')
    device_t=torch.device(device)

    cache_file = 'train_features_{}_{}_{}'.format(f_params['max_query_length'],
                                                  f_params['doc_stride'],
                                                  f_params['max_query_length'])
    if f_params['load_from_cache_train']:
        print('\nLoading train features from cache..\n')
        if os.path.isfile(os.path.join(os.getcwd(),'cache',cache_file)):
            cached_tensor = torch.load(os.path.join(os.getcwd(),'cache',cache_file))
            features, dataset, examples = cached_tensor['features'], cached_tensor['dataset'], cached_tensor['examples']
            print("\nLoaded", len(features),len(dataset),len(examples))
            sub_sample = int(len(examples)*t_params['subset_samples'])
            examples = examples[:sub_sample]
        else:
            raise Exception('Cache File doesnt exist..')
    else:
        processor = SquadProcessor(train_file=ds.train_file, dev_file=ds.test_file)
        examples = processor.get_train_examples(ds.dataset_root,'train-v2.0.json')
        sub_sample = int(len(examples)*t_params['subset_samples'])
        examples = examples[:sub_sample]

        # load features
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        features, dataset = squad_convert_examples_to_features(
                    examples=examples,
                    tokenizer=tokenizer,
                    max_seq_length=f_params['max_seq_length'],
                    doc_stride=f_params['doc_stride'],
                    max_query_length=f_params['max_query_length'],
                    is_training=True)
        # cache the data
        print('\nCaching Train Features..\n')
        if not os.path.exists(os.path.join(os.getcwd(),'cache')):
            os.makedirs(os.path.join(os.getcwd(),'cache'), exist_ok=True)

        torch.save({"features": features, "dataset": dataset, "examples":examples}, 
                   os.path.join(os.getcwd(),'cache',cache_file))

    # move model to the gpu
    if isinstance(mod, str):
        mod = torch.load(mod)
    
    # moving model to device
    mod = mod.to(device_t)

    # loading hyper-parameters
    bs = t_params['batch_size']
    train_epochs = t_params['train_epochs']

    # taking a proportion of total samples
    slice_examples = int(t_params['subset_samples']*len(dataset))
    dataset = Subset(dataset, np.arange(slice_examples))
    train_sampler = RandomSampler(dataset)
    train_dataloader = DataLoader(dataset, sampler=train_sampler, batch_size=bs)


    # Prepare optimizer and schedule (linear warmup and decay)
    gradient_accumulation_steps = t_params['gradient_accum_steps']
    total_train_steps = (len(train_dataloader) // gradient_accumulation_steps) * train_epochs
    no_decay = ["bias", "LayerNorm.weight"]
    weight_decay = 0.0
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in mod.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {"params": [p for n, p in mod.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    if t_params['optimizer']['name'] == 'Adam':
        optimizer = AdamW(optimizer_grouped_parameters, lr=t_params['optimizer']['lr'], eps=1e-8)
    elif t_params['optimizer']['name'] == 'SGD':
        optimizer = SGD(optimizer_grouped_parameters, lr=t_params['optimizer']['lr'], momentum=0.9)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=t_params['warm_up_steps'], num_training_steps=total_train_steps
    )

    # train model
    mod.zero_grad()
    for epoch in range(train_epochs):
        print(f'Epoch-{epoch}')
        for batch in tqdm(train_dataloader, total=len(train_dataloader)):
            
            # model to training mode
            mod.train()
            # print(batch)
            batch = tuple(t.to(device) for t in batch)
            inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                    "start_positions": batch[3],
                    "end_positions": batch[4],
                }
            outputs = mod(**inputs)
            loss = outputs['loss']
            
            # find gradients and update model
            loss.backward()
            optimizer.step()
            scheduler.step()
            mod.zero_grad()
    
    # Saving the model
    if not os.path.exists(os.path.join(os.getcwd(),'output',run_id)):
        os.makedirs(os.path.join(os.getcwd(),'output',run_id), exist_ok=True)
    
    save_path = os.path.join(os.getcwd(),'output',run_id, f'{model_type}_{model_name}.pt')
    print(f'\nSaving model checkpoint.. at {save_path}..\n')
    torch.save(mod, save_path)

    # evaluate model
    inf_rows = QAEvaluate(mod, f_params, e_params, ds, model_type, model_name, run_id, device=device)
    return inf_rows