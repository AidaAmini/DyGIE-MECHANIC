
# COFIE: COVID-19 Open Functional Information Extraction

This repository contains models, datasets and experiments described in [Extracting a Knowledge Base of Mechanisms from COVID-19 Papers](https://arxiv.org/pdf/2010.03824.pdf).

<img src="https://github.com/AidaAmini/DyGIE-COFIE/blob/master/COFIE.png" width="400" height="400"> <img src="https://github.com/AidaAmini/DyGIE-COFIE/blob/master/COFIE-G.png" width="375" height="300" style="vertical-align: top;">

* Please cite our paper if you use our datasets or models in your project. See the [BibTeX](#citation). 
* Feel free to [email us](#contact-us).

# COFIE / COFIE-G datasets
We provide two annotated datasets:
- COFIE: Coarse-grained mechanism relations (`Direct` and `Indirect`)
- COFIE-G: Granular mechanism relations (`Subject-Predicate-Object`)

From project root, run `scripts/data/get_cofie.sh` to download both datasets to the `data` directory.
- `COFIE` will be downloaded to `data/cofie/[train,dev,test].json`. Development and test sets for are also available in tabular format: `data/cofie-gold/[dev,test]-gold.tsv`
- `COFIE-G` will be downloaded to `data/cofie-g/split/[train,dev,test].json`. Tabular format:`data/cofie-g-gold/[dev,test]-gold.tsv`


## Pretrained models
We provide models pre-trained on COFIE and COFIE-G.

### Downloads

From project root, run `scripts/pretrained/get_cofie_pretrained.sh` to download all the available pretrained models to the `pretrained` directory. If you only want one model, here are the download links.

- [Coarse relation prediction model](https://ai2-s2-cofie.s3-us-west-2.amazonaws.com/models/binary-model.tar.gz)
- [Granular relation prediction model](https://ai2-s2-cofie.s3-us-west-2.amazonaws.com/models/ternary-model.tar.gz)

## Table of Contents
- [Dependencies](#dependencies)
- [Making predictions on existing datasets](#making-predictions-on-existing-datasets)
- [Relation extraction evaluation metric](#relation-extraction-evaluation-metric)
- [Training with Allentune](#training-with-allentune)


## Dependencies
This code repository is forked from [DYGIE++](https://github.com/dwadden/dygiepp/blob/allennlp-v1), [Wadden 2019.](https://www.semanticscholar.org/paper/Entity%2C-Relation%2C-and-Event-Extraction-with-Span-Wadden-Wennberg/fac2368c2ec81ef82fd168d49a0def2f8d1ec7d8)

This code was developed using Python 3.7. To create a new Conda environment using Python 3.7, do `conda create --name cofie python=3.7`.

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
allennlp predict pretrained/ternary-model.tar.gz \
    data/cofie-g/split/test.json \
    --predictor dygie \
    --include-package dygie \
    --use-dataset-reader \
    --output-file predictions/cofie-g-test.jsonl \
    --cuda-device 0 \
    --silent
```

For predicting coarse relations using a pretrained model:

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


Running these commands will provide json-formatted predictions.

Alternatively you can use the predict scripts provided by this library to generate both .tsv and .json file. You can use :

```bash
python predict_binary.py --data_dir data/cofie --device 0 --serial_dir pretrained/binary-model.tar.gz  --pred_dir predictions/cofie-test/
```
for coarse relation predictions and

```bash
python predict_ternary.py --data_dir data/cofie-g/collated --device 0 --serial_dir pretrained/ternary-model.tar.gz  --pred_dir predictions/cofie-t-test/
```
for granular relation predictions.

### Relation extraction evaluation metric

We report `Precision/Recall/F1` measured by using exact and partial span-matching functions. Full details are described in our paper.


### Training with Allentune
We use Allentune for hyperparameter tuning. To train a model for coarse relation extraction using Allentune, you can run the script below.

```bash
python scripts/train/train_allentune.py --data_dir data/cofie --device 0,1,2,3 --serial_dir models/cofie/ --gpu_count 4 --cpu_count 12 --device 0,1,2,3
```

To train the model for granular relations:
```bash
python scripts/train/train_event_allentune.py --data_dir data/processed/collated_events/ --serial_dir ./models/events --gpu_count 4 --cpu_count 12 --device 0,1,2,3
```

The default number of training samples is set to 30. For more training options please use the `--h` command.

To obtain predictions for the development set over all Allentune runs:
```bash
python predict.py --data_dir data/cofie --device 0 --serial_dir models/cofie/
```
for the coarse relation model and

```bash
python predict_event_allentune.py --serial_dir ./models/cofie-t --data_dir ./data/cofie-t/ --pred_dir ./predictions/cofie-t
```
for the granular relation model.

You can get test set predcitions by indicating only the run index you want to use for inference:

```bash
python predict.py --data_dir data/cofie --device 0,1,2,3 --serial_dir models/cofie/  --pred_dir predictions/cofie
```
for coarse relations and

```bash
python predict_event_allentune.py --serial_dir ./models/cofie-t --data_dir ./data/cofie-t/ --pred_dir ./predictions/cofie-t --test_data --test_index 17
```
for granular relations.


## Citation

If using our dataset and models, please cite:

```
@inproceedings{amini-hope-2020-cofie,
    title={{Extracting a Knowledge Base of Mechanisms from COVID-19 Papers
}},
    author={Tom Hope and Aida Amini and David Wadden and Madeleine van Zuylen and E. Horvitz and Roy Schwartz and Hannaneh Hajishirzi},
    year={2020},
    url={https://arxiv.org/pdf/2010.03824.pdf}
}
```

## Contact us

Please don't hesitate to reach out.

**Email:** `tomh@allenai.org`, `amini91@cs.washington.edu`

