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

### Inference Exemple 
`$ python main.py -input data/PGxCorpus -output PGxAnnotation_77 -conf_threshold 77'

