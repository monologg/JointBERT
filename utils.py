import os
import random
import logging

import torch
import numpy as np
from seqeval.metrics import precision_score, recall_score, f1_score

from transformers import BertConfig, DistilBertConfig, AlbertConfig
from transformers import BertTokenizer, DistilBertTokenizer, AlbertTokenizer

from model import JointBERT, JointDistilBERT, JointAlbert

MODEL_CLASSES = {
    'bert': (BertConfig, JointBERT, BertTokenizer),
    'bertzh': (BertConfig, JointBERT, BertTokenizer),
    'distilbert': (DistilBertConfig, JointDistilBERT, DistilBertTokenizer),
    'albert': (AlbertConfig, JointAlbert, AlbertTokenizer)
}

MODEL_PATH_MAP = {
    'bert': 'bert-base-uncased',
    'bertzh': 'bert-base-chinese',
    'distilbert': 'distilbert-base-uncased',
    'albert': 'albert-xxlarge-v1'
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


def compute_metrics(intent_preds, intent_labels, slot_preds, slot_labels):
    assert len(intent_preds) == len(intent_labels) == len(slot_preds) == len(slot_labels)
    results = {}
    intent_result = get_intent_acc(intent_preds, intent_labels)
    slot_result = get_slot_metrics(slot_preds, slot_labels)
    sementic_result = get_sentence_frame_acc(intent_preds, intent_labels, slot_preds, slot_labels)

    results.update(intent_result)
    results.update(slot_result)
    results.update(sementic_result)

    return results


def get_slot_metrics(preds, labels):
    assert len(preds) == len(labels)
    return {
        "slot_precision": precision_score(labels, preds),
        "slot_recall": recall_score(labels, preds),
        "slot_f1": f1_score(labels, preds)
    }


def get_intent_acc(preds, labels):
    acc = (preds == labels).mean()
    return {
        "intent_acc": acc
    }


def read_prediction_text(args):
    return [text.strip() for text in open(os.path.join(args.pred_dir, args.pred_input_file), 'r', encoding='utf-8')]


def get_sentence_frame_acc(intent_preds, intent_labels, slot_preds, slot_labels):
    """For the cases that intent and all the slots are correct (in one sentence)"""
    # Get the intent comparison result
    intent_result = (intent_preds == intent_labels)

    # Get the slot comparision result
    slot_result = []
    for preds, labels in zip(slot_preds, slot_labels):
        assert len(preds) == len(labels)
        one_sent_result = True
        for p, l in zip(preds, labels):
            if p != l:
                one_sent_result = False
                break
        slot_result.append(one_sent_result)
    slot_result = np.array(slot_result)

    sementic_acc = np.multiply(intent_result, slot_result).mean()
    return {
        "sementic_frame_acc": sementic_acc
    }



enSet = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-\'")
digitSet = set("1234567890")
pinyinSet = set("āáǎàōóǒòēéěèīíǐìūúǔùüǖǘǚǜ")
def is_zh(c):
    x = ord (c)
    if x >= 0x4e00 and x <= 0x9fbb:
        return True
    # CJK Compatibility Ideographs
    elif x >= 0xf900 and x <= 0xfad9:
        return True
    # CJK Unified Ideographs Extension B
    elif x >= 0x20000 and x <= 0x2a6d6:
        return True
    # CJK Compatibility Supplement
    elif x >= 0x2f800 and x <= 0x2fa1d:
        return True
    else:
        return False

def alphabet2typeArray(alphabet):
    """标记输入字符串每个字符的类型"""
    type1 = np.empty([len(alphabet)], np.int32)
    for i, c in enumerate(alphabet):
        if is_zh(c):
            type1[i]=0       #中文
        elif c in enSet:
            type1[i]=1       #英文
        elif c in pinyinSet:
            type1[i]=1       #拼音
        elif c in digitSet:
            type1[i]=2       #数字
        elif c=="\u3000" or c==" ":
            type1[i]=3       #空格
        else:
            type1[i]=4       #符号
    return type1

def split_Mix_word(str1):
    """对混合字符串进行分词，中文字符分，英文和数字合并，符合单独"""
    typeList = alphabet2typeArray(str1) # 0:zh, 1:en, 2:digst 3:space 4:symbol
    
    words = []      # 存放分割好的词,字
    prevtype = -1   # 标记上一次的type
    tmpword = ""    # 记录需要合并的字符
    for i, (char, type1) in enumerate(zip(str1, typeList)):
        if tmpword and prevtype != type1 and not(prevtype==1 and type1==2):
            words.append(tmpword)
            tmpword = ""; prevtype = type1
        if type1==0:
            words.append(char)
        elif type1==1 or type1==2:
            tmpword += char
            prevtype = type1
        elif type1 ==3: continue
        elif type1 == 4: words.append(char)
        if i==len(str1)-1 and tmpword: words.append(tmpword)
    return words

if __name__ == "__main__":
    # 中英混合分词测试
    strlist = [
        # "play the top20 best chicane songs on deezer",
        "add the entire album into indie español"
        # "In this page, we will show you how to share a model增加样本#数量12",
        # "请问In this page, we will show you how to share a model",
        # "12加14等于多少?",
        # "我喜欢旅游"
        ]
    for str1 in strlist:
        words = split_Mix_word(str1)
        print(words)