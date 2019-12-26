import os
import random
import logging

import torch
import numpy as np
from transformers import BertTokenizer, BertConfig, DistilBertConfig, DistilBertTokenizer, RobertaConfig, RobertaTokenizer, \
    AlbertConfig, AlbertTokenizer

from model import JointBERT, JointDistilBERT

MODEL_CLASSES = {
    'bert': (BertConfig, JointBERT, BertTokenizer),
    'distilbert': (DistilBertConfig, JointDistilBERT, DistilBertTokenizer),
    'roberta': (RobertaConfig, JointBERT, RobertaTokenizer),
    'albert': (AlbertConfig, JointBERT, AlbertTokenizer)
}

MODEL_PATH_MAP = {
    'bert': 'bert-base-uncased',
    'distilbert': 'distilbert-base-uncased',
    'roberta': 'roberta-base',
    'albert': 'albert-large-v1'
}


def get_intent_labels(args):
    return [label.strip() for label in open(os.path.join(args.data_dir, args.task, args.intent_label_file), 'r', encoding='utf-8')]


def get_slot_labels(args):
    return [label.strip() for label in open(os.path.join(args.data_dir, args.task, args.slot_label_file), 'r', encoding='utf-8')]


def load_tokenizer(args):
    return MODEL_CLASSES[args.model_type][2].from_pretrained(args.model_name_or_path)


def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def compute_metrics(preds, labels):
    assert len(preds) == len(labels)
    return get_acc(preds, labels)


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def get_acc(preds, labels, average='macro'):
    acc = simple_accuracy(preds, labels)
    return {
        "intent_acc": acc,
    }


def read_prediction_text(args):
    return [text.strip() for text in open(os.path.join(args.pred_dir, args.pred_input_file), 'r', encoding='utf-8')]
