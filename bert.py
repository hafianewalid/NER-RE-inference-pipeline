from collections import OrderedDict
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertConfig
from transformers import AutoTokenizer, AutoModel, AutoConfig
from pytorch_transformers import BertModel as bm


def get_bert(bert_type='bert'):
    tokenizer, model = None, None
    if (bert_type == 'bert'):
        ######## bert ###########

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')

        ########################

    if (bert_type == 'biobert'):
        #### Bio BERT #########
        '''
        config = BertConfig(vocab_size_or_config_json_file="biobert_model/bert_config.json")

        model = BertModel(config)
        tmp_d = torch.load("biobert_model/pytorch_model.bin")
        state_dict = OrderedDict()
        for i in list(tmp_d.keys())[:199]:
            x = i
            if i.find('bert') > -1:
                x = '.'.join(i.split('.')[1:])
            state_dict[x] = tmp_d[i]
            '
        #print(state_dict)
        model.load_state_dict(tmp_d)
        tokenizer=BertTokenizer(vocab_file="biobert_model/vocab.txt", do_lower_case=True)
        '''

        model = bm.from_pretrained('biobert_v1.1_pubmed')
        tokenizer = BertTokenizer(vocab_file="biobert_v1.1_pubmed/vocab.txt", do_lower_case=True)

        #### Bio BERT #########

    if (bert_type == 'scibert'):
        #### sci bert #########


        config = AutoConfig.from_pretrained('allenai/scibert_scivocab_uncased', output_hidden_states=True)
        tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
        model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased', config=config)

        #######################

    return tokenizer, model