import random

import torch
import numpy as np

from datasets import load_dataset
from transformers import AutoTokenizer


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def get_wikitext2(nsamples, seed, seqlen, model):
    traindata = load_dataset("arrow",
                            data_files="../datasets/wikitext2_raw_v1/wikitext-train.arrow", 
                            split="train")
    testdata = load_dataset("arrow", 
                            data_files="../datasets/wikitext2_raw_v1/wikitext-test.arrow",
                            split="train")
    
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_ptb(nsamples, seed, seqlen, model):
    traindata = load_dataset("arrow", 
                            data_files="../datasets/penn_treebank/ptb_text_only-train.arrow", 
                            split="train")
    testdata = load_dataset("arrow", 
                            data_files="../datasets/penn_treebank/ptb_text_only-test.arrow", 
                            split="train")

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    trainenc = tokenizer("\n\n".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['sentence']), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_c4(nsamples, seed, seqlen, model):
    traindata = load_dataset("arrow", 
                            data_files="../datasets/allenai_c4/c4-train-0000*-of-00002.arrow", 
                            split="train")
    valdata = load_dataset("arrow", 
                            data_files="../datasets/allenai_c4/c4-validation.arrow", 
                            split="train")

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]

    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids
    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc


def get_loaders(name, nsamples=128, seed=0, seqlen=2048, model=''):
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, model)
    if 'ptb' in name:
        return get_ptb(nsamples, seed, seqlen, model)
    if 'c4' in name:
        return get_c4(nsamples, seed, seqlen, model)
    

if __name__ == "__main__":
    for dataset in ["wikitext2", "ptb", "c4"]:
        get_loaders(dataset, model='../models/llama-3.2-1B')