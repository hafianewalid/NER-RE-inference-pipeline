"""

This module offering word embadding using BERT model

"""

import torch
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')

def Text2tokens(text):
    marked_text = "[CLS] " + text +" [SEP]"
    tokenized_text = tokenizer.tokenize(marked_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [0] * len(tokenized_text)
    return indexed_tokens,segments_ids

def Token2tonsor(token):
    indexed_tokens,segments_ids = token[0],token[1]
    tokens_tensor = torch.tensor([indexed_tokens])
    return tokens_tensor


def get_bert_inputs(text):
    tokenized_id = Text2tokens(text)
    inputs=Token2tonsor(tokenized_id)[0].view(-1)
    return inputs

def get_bert_re_inputs(text):
    tokenized_id = Text2tokens(text)
    inputs = Token2tonsor(tokenized_id)[0].view(-1)

    tokenized_text = tokenizer.tokenize("[CLS] " + text + " [SEP]")
    outind = [0, 0, 0, 0]
    j = 0
    v = False
    u = False

    v1 = False
    u1 = False

    for i in tokenized_text:
        #print(i)
        if ('<' in i and v and outind[0] == 0):
            outind[0] = j - 1
        if ('[' in i and u and outind[2] == 0):
            outind[2] = j - 1
        v = '<' in i
        u = '[' in i

        if ('>' in i and v1 and outind[1] == 0):
            outind[1] = j

        if (']' in i and u1 and outind[3] == 0):
            outind[3] = j
        v1 = '>' in i
        u1 = ']' in i

        j += 1

    if (outind[3] + outind[2] == 0):
        outind[3] = outind[1]
        outind[2] = outind[0]


    return inputs, torch.tensor(outind)