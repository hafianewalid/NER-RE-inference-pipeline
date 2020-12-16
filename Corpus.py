import glob
import random
import re
import string
import jsonlines
import csv

from Embedding import tokenizer, get_bert_inputs, get_bert_re_inputs

Entites=['Chemical','Genomic_factor','Gene_or_protein','Genomic_variation'
        ,'Limited_variation','Haplotype','Phenotype','Disease',
         'Pharmacodynamic_phenotype','Pharmacokinetic_phenotype']

Relation_types=['increases', 'isEquivalentTo','treats','decreases','influences','causes','isAssociatedWith']

def load_txt(txt_path):
    '''
    :param txt_path: path to file
    :return: the text inside the file
    '''
    txt = open(txt_path).read().replace('\n',"")
    return txt


def load_ann(ann_path):
    '''
    :param ann_path: path to annotation file, this file must be in BRAT format
    :return: tuple contains the first entity, the second one and the label according the relationship
    '''
    ann = open(ann_path).read().split('\n')
    labels = []
    T = [i for i in ann if i.startswith('T')]
    for i in T:

        label = Entites.index(i.split()[1])+1
        end_i=3
        start=i.split()[2]
        end=i.split()[end_i]
        while(';' in end):
            end_i+=1
            end = i.split()[end_i]

        range = [int(start), int(end.split(';')[0])]
        labels.append({ 'range':range, 'type':label, 'arg':i.split()[0]})


    return labels


def tokens_ranges(tokens,sent):
    ranges=[]
    p=-1
    for token in tokens:
        if token[0] in string.punctuation:
            p+=" " in sent[p:p+len(token.replace("#", ''))]
            ranges.append([p,p+len(token.replace("#", ''))])
            p-=(p+1<len(sent) and token in string.punctuation and sent[p+1]!=" ")
            p+=len(token.replace("#",''))
        else:
            p+=1
            ranges.append([p,p+len(token)])
            p+=len(token)

    return ranges


def load_data_ner(path):
    files=[(f.split(path)[1]).split('.txt')[0] for f in glob.glob(path +"/*.txt")]
    data=[]
    for id in files:
      sentence = load_txt(path+id+'.txt')
      tokens = tokenizer.tokenize(sentence)
      tr =tokens_ranges(tokens,sentence)
      data.append({'id':id, 'sent': sentence, 'input': get_bert_inputs(sentence),'tokens':tokens , 'tr': tr })
    return data

def copy_txt(src,dist):
    files = [f.split(src)[1] for f in glob.glob(src+ "/*.txt")]
    for id in files:
        sentence = load_txt(src+id)
        file = open(dist+id, "w")
        print(sentence,file=file)
        file.close()


def load_data_re(path):
    ann_txt_files = [(f.split(path)[1], (f.split(path)[1]).split('ann')[0] + "txt") for f in glob.glob(path + "/*.ann")]

    data = []

    for ann, txt in ann_txt_files:

        sentence = load_txt(path + txt)
        entities = load_ann(path + ann)

        for e1 in entities:
            for e2 in entities:
                if e1['arg']!=e2['arg']:
                    sent_enc = sentence[:e1['range'][0]] + "<<" + sentence[e1['range'][0]:e1['range'][1]] + ">>" + sentence[
                                                                                                                   e1[
                                                                                                                       'range'][
                                                                                                                       1]:]
                    shift= 2+(e1['range'][0]<=e2['range'][1])*2 if e1['range'][0]<e2['range'][0] or e1['range'][0]<e2['range'][1] else 0
                    sent_enc = sent_enc[:e2['range'][0]+shift] + "[[" + \
                               sent_enc[e2['range'][0]+shift:e2['range'][1]+shift-(e2['range'][1]<e1['range'][1])] + "]]" + sent_enc[
                                                                                                                   e2[
                                                                                                                       'range'][
                                                                                                                       1]+shift-(e2['range'][1]<e1['range'][1]):]
                    data.append({'id': ann, 'sent': sent_enc, 'input': get_bert_re_inputs(sent_enc), 'e1': e1, 'e2': e2})

    return data


def ner_annotation(data,path,threshold=0):


    for x in data:
        label=x["label"]
        tr=x["tr"]
        sent=x["sent"]
        tokens=x["tokens"]
        conf=x["confidence"]
        e,start,end=-1,0,0
        seg=[]

        # conf filter
        conf_select=[c>=threshold for c in conf]
        label=[l if c else 0 for c,l in zip(conf_select,label)]

        # dilatation
        for i in range(1,len(tokens)-1):
            if(label[i]!=0):
                r,j = tr[i],i
                while " " not in sent[r[0]:r[1]] and j>-1 :
                    label[j]=label[i]
                    j-=1
                    r[0]=tr[j][0]
                r,j = tr[i],i
                while " " not in sent[r[0]:r[1]] and j<len(label)-1 :
                    label[j]=label[i]
                    j+=1
                    r[1]=tr[j][1]


        for i in range(len(tr)):
            token_label=label[i]
            if(token_label!=e):
                seg.append((e,start,end))
                start=tr[i][0]
            e,end=token_label,tr[i][1]
        seg.append((e, start, end))

        t=1
        for i in range(len(seg)):
            if seg[i][0]>0:
                file = open(path + x["id"] + ".ann", "a+")
                print("T"+str(t)+"\t"+Entites[seg[i][0]-1]+"\t"+str(seg[i][1])+
                      " "+str(seg[i][2])+"\t"+sent[seg[i][1]:seg[i][2]],file=file)
                file.close()
                t+=1




def re_annotation(data,path,threshold=0):

    rel={}
    for x in data:
        label = x["label"]
        e1,e2= x["e1"],x["e2"]
        id = x["id"]
        conf = x["confidence"]

        if conf>= threshold:
            if id not in rel.keys():
                rel[id]=0
            rel[id]+=1

            file = open(path + x["id"], "a+")
            print("R"+str(rel[id])+"\t"+Relation_types[label]+"\t"+
                  "Arg1:"+str(e1['arg'])+"\t" +"Arg2:"+str(e2['arg']), file=file)
            file.close()



