import os

import argparse
import  torch
from tqdm import tqdm
from Corpus import load_data_ner, ner_annotation, copy_txt, load_data_re, re_annotation

parser = argparse.ArgumentParser()

parser.add_argument('-input', default='data/PGxCorpus', type=str,
                            dest='data_path',
                            help='Inputs data path, text files (.txt)')

parser.add_argument('-output', default='Annotation', type=str,
                            dest='outputs_path',
                            help='Outputs data path, NER and RE annotations files in brat format (.ann)')

parser.add_argument('-conf_threshold', default=0, type=int, choices=range(0,101),
                            dest='c_threshold',
                            help='Threshold on confidence prediction')


param = parser.parse_args()
data_path= param.data_path
output_path= param.outputs_path
c_threshold=param.c_threshold*0.01

use_gpu = torch.cuda.is_available()
if use_gpu:
    print("°°°°°°°°°°°°°°°°°°°°  GPU  °°°°°°°°°°°°°°°°°°")
    device = torch.device('cuda')
else:
    print("°°°°°°°°°°°°°°°°°°°°  CPU  °°°°°°°°°°°°°°°°°°")
    device = torch.device('cpu')


NER_model=torch.load("NER.pt",map_location=device)
print(NER_model)

ner_data = load_data_ner(data_path)
pbar = tqdm(total=len(ner_data), desc=" NER annotation : ")

for x in ner_data:
    input = torch.stack([x["input"]])
    input = input.to(device)
    with torch.no_grad():
        NER_model.eval()
        output=NER_model(input)
        predicted_targets = output[0].argmax(dim=2)
        confidence = output[0].max(dim=2).values
        x["label"]=(predicted_targets.tolist()[0])[1:-1]
        x["confidence"]=(confidence.tolist()[0])[1:-1]
    pbar.update(1)
    continue
pbar.close()

if not os.path.exists(output_path):
    os.makedirs(output_path)


ner_annotation(ner_data,output_path,threshold=c_threshold)



RE_model=torch.load("RE.pt",map_location=device)
print(RE_model)

copy_txt(data_path,output_path)
re_data=load_data_re(output_path)

pbar = tqdm(total=len(re_data), desc=" RE annotation : ")
for x in re_data:
    input1 = torch.stack([x["input"][0]])
    input2 = torch.stack([x["input"][1]])
    input1,input2=input1.to(device),input2.to(device)
    with torch.no_grad():
        RE_model.eval()
        output = RE_model(input1,input2)
        predicted_targets = output.argmax(dim=1)
        confidence = output.max(dim=1).values
        x['label'] = predicted_targets.tolist()[0]
        x["confidence"] = confidence.tolist()[0]
        pbar.update(1)
        continue
pbar.close()

re_annotation(re_data,output_path,threshold=c_threshold)
