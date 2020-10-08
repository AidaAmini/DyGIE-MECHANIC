
# COFIE: COVID-19 Open Functional Information Extraction

This repository contains models, datasets and experiments described in [Extracting a Knowledge Base of Mechanisms from COVID-19 Papers](TBA)

# COFIE / COFIE-G datasets
We provide two annotated datasets:
- COFIE: Coarse-grained mechanism relations (`Direct` and `Indirect`)
- COFIE-G: Granular mechanism relations (`subject-predicate-object`)

![COFIE](https://github.com/AidaAmini/DyGIE-COFIE/blob/master/COFIE.png)
![COFIE-G](https://github.com/AidaAmini/DyGIE-COFIE/blob/master/COFIE-G.png)


COFIE is available in data/cofie/[train,dev,test].json. The gold labeled data of dev and test sets for evaluation is in cofie-gold/[dev,test]-gold.tsv
COFIE-G is in data/cofie-t/split/[train,dev,test].json. Gold labels are in cofie-t-gold/[dev,test]-gold.tsv


# DyGIE++COIFE


## Table of Contents
- [Dependencies](#dependencies)
- [Pretrained models](#Pretrained-models)
- [Downloads](#downloads)
- [Making predictions on existing datasets](#making-predictions-on-existing-datasets)
- [Relation extraction evaluation metric](#relation-extraction-evaluation-metric)
- [Training with Allentune](#training-with-allentune)
- [Running prediction for models trained with Allentune](#running-prediction-for-models-trained-with-allentune)
- [Contact](#contact)


## Dependencies
This code repository is forked from [DYGIE++](https://github.com/dwadden/dygiepp/blob/allennlp-v1), [Wadden 2019.](https://www.semanticscholar.org/paper/Entity%2C-Relation%2C-and-Event-Extraction-with-Span-Wadden-Wennberg/fac2368c2ec81ef82fd168d49a0def2f8d1ec7d8)

This code was developed using Python 3.7. To create a new Conda environment using Python 3.7, do `conda create --name cofie python=3.7`.

For installing compatible Allentune library, please download the allentune git repo outside of dygiepp directory using : `git clone https://github.com/allenai/allentune.git`, then replace the files provided in this repository using command `cp -r allentune_files/ [location of downloaded allentune]`. The you can proceed with installing allentune by running `pip install --editable .` in allentune downloaded folder.

After installing allentune please proceed with installing required libraries for DyGIE++.

The necessary dependencies can be installed with `pip install -r requirements.txt`.

This library relies on [AllenNLP](https://allennlp.org) and uses AllenNLP shell [commands](https://docs.allennlp.org/master/#package-overview) to kick off training, evaluation, and testing.


## Pretrained models 
We have the models trained on COFIE (binary relations) and COFIE-t (ternary relations) data available. 

### Downloads 

Run `scripts/pretrained/get_cofie_pretrained.sh` to download all the available pretrained models to the `pretrained` directory. If you only want one model, here are the download links.

- [Binary relation prediction model](https://ai2-s2-cofie.s3-us-west-2.amazonaws.com/models/binary-model.tar.gz)
- [Ternary relation prediction model](https://ai2-s2-cofie.s3-us-west-2.amazonaws.com/models/ternary-model.tar.gz)


## Making predictions on existing datasets 
-- option[1]
To make a prediction, you can use `allennlp predict`. For example, to make a prediction with the pretrained scierc model, you can do:

```bash
allennlp predict pretrained/ternary-model.tar.gz \
    data/cofie/split//test.json \
    --predictor dygie \
    --include-package dygie \
    --use-dataset-reader \
    --output-file predictions/cofie-t-test.jsonl \
    --cuda-device 0 \
    --silent
```
for ternary relation prediction 

```bash
allennlp predict pretrained/binary.tar.gz \
    data/cofie/test.json \
    --predictor dygie \
    --include-package dygie \
    --use-dataset-reader \
    --output-file predictions/cofie-test.jsonl \
    --cuda-device 0 \
    --silent
```

for binary relation prediction.

Running these command will provide the json formatted prediction. 
-- option [2]
Alternatively you can use the predict scripts provided by this library to generate both .tsv and .json file. You can use :

```bash
python predict_binary.py --data_dir data/cofie --device 0 --serial_dir pretrained/binary-model.tar.gz  --pred_dir predictions/cofie-test/
```
for binary relation predictions and 

```bash
python predict_ternary.py --data_dir data/cofie-t/collated --device 0 --serial_dir pretrained/ternary-model.tar.gz  --pred_dir predictions/cofie-t-test/
```
for ternary relation predictions and 


### Relation extraction evaluation metric 

We report P/R/F1 metrics measured by using exact and partial matching metrics. A relation is selected as correct if both sides of a relation match by the chosen metric with a relation from gold set.
The relaton metircs are :
--Exact match
--Substring match
--Rouge similarity metric

All the entities are from the same general entity type, therefore there is not label matching in entity identification.

For binary relation task : We report the scores of two task of relation identification and relation classification. In relation identification, we check for spans on both sides of relations. In addition, in relation classification task we consider the relation label.
For ternary relation task : We consider the third argument as label of the relation and therefore we report relation identification and relation classification tasks. 


### Training with Allentune 
By using Allentune librrary alongside with Dygie, we are able to implement a greed search for hyperparameter tuning. In our experince the using Allentune can significantly improve the results.
To train a model for binary relation prediction using Allentune, you can run the script below. 

```bash 
python scripts/train/train_allentune.py --data_dir data/cofie --device 0,1,2,3 --serial_dir models/cofie/ --gpu_count 4 --cpu_count 12 --device 0,1,2,3
```

To train the model for ternary relation prediction you can run :
```bash
python scripts/train/train_event_allentune.py --data_dir data/processed/collated_events/ --serial_dir ./models/events --gpu_count 4 --cpu_count 12 --device 0,1,2,3
```


The default number of training samples is set to 30. For more traning option please use --h command.


### Running prediction for models trained with Allentune 
You can use the script provided in this library to predict the relation for all the runs saved in a directory for development set and specify a best training index for predicting over test set once you find the best training parameters.
To run the prediction for development set over all the trained models you can run :
```bash 
python predict.py --data_dir data/cofie --device 0 --serial_dir models/cofie/ 
```
for binary relation model and 

```bash 
python predict_event_allentune.py --serial_dir ./models/cofie-t --data_dir ./data/cofie-t/ --pred_dir ./predictions/cofie-t 
```
for ternary relation model.


Once you have the best training index, you can get the test set predcitions by indicating only the index you want to get the prediction for. Now you can run 

```bash 
python predict.py --data_dir data/cofie --device 0,1,2,3 --serial_dir models/cofie/  --pred_dir predictions/cofie
```
for binary relation model and 

```bash 
python predict_event_allentune.py --serial_dir ./models/cofie-t --data_dir ./data/cofie-t/ --pred_dir ./predictions/cofie-t --test_data --test_index 17
```
for ternary relation model.


