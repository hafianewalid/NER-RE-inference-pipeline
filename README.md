# Named Entity Recognition(NER)- Relation Extraction (RE) - inference pipeline
## Run an inference for a text dataset
For using the pipeline (NER-RE) run the main code on an CPU or GPU device with the according parameters.

### Display parameters list :

`$ python main.py -h`

### Input Data :
The input data should be one derectory in .txt files format.

### Output Annotation :
The output annotation will be created in the brat format.

### Parameters 
* input : the inputs data derectory
* output : the outputs annotations derectory
* conf_threshold : the confidence filtre threshold,within 0%-100%

### Download Models

NER-Model (SciBERT uncased token classifier for NER)

`$ gdown https://drive.google.com/uc?id=1nwq1BLRv5lruhA0R_jerm2OGxQGnFGx7`

RE-Model (SciBERT-MCNN-segmentation for RE)

`$ gdown https://drive.google.com/uc?id=1XB0AriPdZUNiJF4FJ3B2TR1Jfq0nWPVc`

### Inference Example 

`$ python main.py -input data/PGxCorpus -output PGxAnnotation_77 -conf_threshold 77`

For exemple the annotation for the file 10585223_5.txt is in 10585223_5.ann file and should be as follows :

```
T1	Pharmacokinetic_phenotype	0 33	Brain concentration-time profiles
T2	Limited_variation	37 84	mdr1a ( + / + ) and mdr1a ( - / - ) mice showed
T3	Pharmacokinetic_phenotype	160 180	oxin accumulation in
R1	isAssociatedWith	Arg1:T1	Arg2:T2
R2	isAssociatedWith	Arg1:T1	Arg2:T3
R3	influences	Arg1:T2	Arg2:T1
R4	influences	Arg1:T2	Arg2:T3
R5	influences	Arg1:T3	Arg2:T1
R6	isAssociatedWith	Arg1:T3	Arg2:T2
```

