from copy import copy

import numpy as np
from evaluate import load
from datasets import load_dataset, Dataset


feautre_mappings = {'sentence': 'text', 'polarity': 'label'}
label_mappings = {-1: 0, 0: 1, 1: 2}
dataset_id = "fhamborg/news_sentiment_newsmtsc"

def compute_metrics(eval_pred):
   load_f1 = load("f1")
   logits, labels = eval_pred
   predictions = np.argmax(logits, axis=-1)
   f1 = load_f1.compute(predictions=predictions, references=labels, average='macro')["f1"]
   return {"f1": f1}

def preprocess_label(input):
   input = copy(input)
   input['label'] = label_mappings[input['label']]
   return input

def get_dataset(tokenizer = None) -> Dataset:
    dataset = load_dataset(dataset_id)
    unused_feauters = list(set(dataset.column_names['train']) - set(feautre_mappings.keys()))
    dataset =  dataset.remove_columns(unused_feauters)
    dataset = dataset.rename_columns(feautre_mappings)

    dataset = dataset.map(preprocess_label)
    if tokenizer:
        tokenize = lambda input: tokenizer(input["text"], truncation=True, padding='max_length', max_length=512)
        dataset = dataset.map(tokenize, batched=True)
    return dataset