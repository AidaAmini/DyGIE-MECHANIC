
We introduce a novel schema for `mechanisms` that generalizes across many types of activities, functions and influences in scientific literature. This repository contains models, datasets and experiments described in our NAACL 2021 paper: [Extracting a Knowledge Base of Mechanisms from COVID-19 Papers](https://arxiv.org/pdf/2010.03824.pdf).

<img src="https://github.com/AidaAmini/DyGIE-MECHANIC/blob/master/teaser.png" width="500" height="400"> <img src="https://github.com/AidaAmini/DyGIE-MECHANIC/blob/master/COFIE-G.png" width="375" height="300" style="vertical-align: top;">

* Please cite our paper if you use our datasets or models in your project. See the [BibTeX](#citation). 
* Feel free to [email us](#contact-us).

# Annotated datasets
We provide two annotated datasets:
- Coarse-grained mechanism relations (`Direct` and `Indirect`)
- Granular mechanism relations (`Subject-Predicate-Object`)

From project root, run `scripts/data/get_mechanic.sh` to download both datasets to the `data` directory.
- Coarse-grained relations will be downloaded to `data/mechanic/coarse/[train,dev,test].json`. Development and test sets for are also available in tabular format: `data/mechanic/coarse-gold/[dev,test]-gold.tsv`
- Granular relations will be downloaded to `data/mechanic/granular/[train,dev,test].json`. Tabular format:`data/mechanic/granular-gold/[dev,test]-gold.tsv`


## Pretrained models
We provide models pre-trained on both datasets.

### Downloads

From project root, run `scripts/pretrained/get_mechanic_pretrained.sh` to download all the available pretrained models to the `pretrained` directory. If you only want one model, here are the download links.

- [Coarse relation prediction model](https://s3-us-west-2.amazonaws.com/ai2-s2-mechanic/models/mechanic-coarse.tar.gz)
- [Granular relation prediction model](https://s3-us-west-2.amazonaws.com/ai2-s2-mechanic/models/mechanic-granular.tar.gz)

## Table of Contents
- [Dependencies](#dependencies)
- [Making predictions on existing datasets](#making-predictions-on-existing-datasets)
- [Relation extraction evaluation metric](#relation-extraction-evaluation-metric)
- [Training with Allentune](#training-with-allentune)


## Dependencies
This code repository is forked from [DYGIE++](https://github.com/dwadden/dygiepp/blob/allennlp-v1), [Wadden 2019.](https://www.semanticscholar.org/paper/Entity%2C-Relation%2C-and-Event-Extraction-with-Span-Wadden-Wennberg/fac2368c2ec81ef82fd168d49a0def2f8d1ec7d8)

This code was developed using Python 3.7. To create a new Conda environment using Python 3.7, do `conda create --name mechanic python=3.7`.

This library relies on [AllenNLP](https://allennlp.org) and uses AllenNLP shell [commands](https://docs.allennlp.org/master/#package-overview) to kick off training, evaluation, and testing.

We use the [Allentune](ttps://github.com/allenai/allentune) for hyperparameter search. For installing a compatible version of the Allentune library, please download the allentune git repo outside of dygiepp directory using:
```bash
git clone https://github.com/allenai/allentune.git
```
Then replace the files provided in this repository using command
```bash
cp -r allentune_files/[location of downloaded allentune]
```
The you can proceed with installing allentune by running
```
pip install --editable .
```
in allentune downloaded folder.

After installing allentune please proceed with installing required libraries for DyGIE++. The necessary dependencies can be installed with
```
pip install -r requirements.txt
```


## Making predictions on existing datasets

To make a prediction, you can use `allennlp predict`. For example, to make a prediction with a pretrained granular relation model:

```bash
allennlp predict pretrained/mechanic-granular.tar.gz \
    data/mechanic/granular/test.json \
    --predictor dygie \
    --include-package dygie \
    --use-dataset-reader \
    --output-file predictions/granular-test.jsonl \
    --cuda-device 0 \
    --silent
```

For predicting coarse relations using a pretrained model:

```bash
allennlp predict pretrained/mechanic-coarse.tar.gz \
    data/mechanic/coarse/test.json \
    --predictor dygie \
    --include-package dygie \
    --use-dataset-reader \
    --output-file predictions/coarse-test.jsonl \
    --cuda-device 0 \
    --silent
```


Running these commands will provide json-formatted predictions.

Alternatively you can use the predict scripts provided by this library to generate both .tsv and .json file. You can use :

```bash
python predict_coarse.py --data_dir data/mechanic/coarse --device 0 --serial_dir pretrained/mechanic-coarse.tar.gz  --pred_dir predictions/coarse-test/
```
for coarse relation predictions and

```bash
python predict_granular.py --data_dir data/mechanic/granular --device 0 --serial_dir pretrained/mechanic-granular.tar.gz  --pred_dir predictions/granular-test/
```
for granular relation predictions.

### Relation extraction evaluation metric

We report `Precision/Recall/F1` measured by using exact and partial span-matching functions. Full details are described in our paper.


### Training with Allentune
We use Allentune for hyperparameter tuning. To train a model for coarse relation extraction using Allentune, you can run the script below.

```bash
python scripts/train/train_coarse_allentune.py --data_dir data/mechanic/coarse/ --device 0,1,2,3 --serial_dir models/coarse/ --gpu_count 4 --cpu_count 12 --device 0,1,2,3
```

To train the model for granular relations:
```bash
python scripts/train/train_granular_allentune.py --data_dir data/mechanic/granular/ --serial_dir ./models/granular --gpu_count 4 --cpu_count 12 --device 0,1,2,3
```

The default number of training samples is set to 30. For more training options please use the `--h` command.

To obtain predictions for the development set over all Allentune runs:
```bash
python predict_coarse_allentune.py --data_dir data/mechanic/coarse/ --device 0 --serial_dir models/coarse/ --pred_dir predictions/coarse
```
for the coarse relation model and

```bash
python predict_granular_allentune.py --serial_dir ./models/granular --data_dir ./data/mechanic/granular/ --pred_dir ./predictions/granular/
```
for the granular relation model.

You can get test set predcitions by indicating only the run index you want to use for inference:

```bash
python predict_coarse_allentune.py --data_dir data/mechanic/coarse/ --device 0,1,2,3 --serial_dir models/coarse/  --pred_dir predictions/coarse
```
for coarse relations and

```bash
python predict_granular_allentune.py --serial_dir ./models/granular --data_dir ./data/mechanic/granular/ --pred_dir ./predictions/granular/ --test_data --test_index 17
```
for granular relations.


## Citation

If using our dataset and models, please cite:

```
@inproceedings{mechanisms21,
    title={{Extracting a Knowledge Base of Mechanisms from COVID-19 Papers
}},
    author={Tom Hope and Aida Amini and David Wadden and Madeleine van Zuylen and Sravanthi Parasa and Eric Horvitz and Daniel Weld and Roy Schwartz and Hannaneh Hajishirzi},
    year={2021},
    journal={NAACL}
}
```

## Contact us

Please don't hesitate to reach out.

**Email:** `tomh@allenai.org`, `amini91@cs.washington.edu`

