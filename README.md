# Task Specific Model Finetuning Framework

A modular framework for task-specific finetuning of pre-trained language models. Supports NLU (Natural Language Understanding) tasks such as Question Answering and Sentence Similarity. Configs can be used to define tasks, base models (from hugging face), the dataset class for finetuning and other training and evaluation hyperparameters. Driver files are used to launch training/evaluation jobs based on pre-defined configs. The framework is easily extendable to other NLU tasks.

## Running the code
```
# From root directory install all the dependancies
pip install -r requirements.txt

# set configs (configs/run_config.json) and launch tool
./driver/driver.py
```

## Directory Structure

```
│───README.md
│───requirements.txt
│   
├───configs
│   └───run_config.json
│   └───run_config_appendix.json
│
├───dataset
│   |───dataset.py
│   │
│   ├───STSRawData
│   │   └───sts2016-english-with-gs-v1.0
│   │           correlation-noconfidence.pl
│   │           LICENSE.txt
│   │           README.txt
│   │           STS2016.gs.answer-answer.txt
|   |           .......
|   |           .......
├───driver
│   └───driver.py
|   └───driver_ensemble.py
│
├───models
│   └───model.py
│   └───model_maps.py
├───scripts
│   └───correlation-noconfidence.pl
│
└───utils
    └───evaluation.py
    └───utility.py
```

## Files
1. `driver/driver.py`: Starting point using which tool is launched.
2. `models/model.py`: Model definitions (Sentence Transformer, Word2vec, Doc2Vec, Glove, and others).
3. `configs/run_config_sts.json`: Specify the model names and device to run the inference (CPU  ("cpu") or GPU ("cuda") supported). If Dataset download is set to false, existing dataset folder used under dataset/STSRawData, otherwise dataset is fetched from given download link.
4. `models/model_maps.py`: Supported models based on libraries offering their pre-trained weights. To Extend the package, this file needs to be augmented.

## Ensemble Files
1. `driver/driver_ensemble.py`: Starting point using the tool to train ensemble on SQUAD dataset.
2. `train`: train files and utilties for squad and sts
3. `evaluation`: evaluation files and utilities for squad and sts
4. `models`: QA and STS model variations and their definitions
5. `configs/run_config_squad.json`: Specify the model names and device to run the inference (CPU  ("cpu") or GPU ("cuda") supported). If Dataset download is set to false, existing dataset 

