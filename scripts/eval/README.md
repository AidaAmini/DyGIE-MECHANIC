# KB creation 

**USABILITy**: This directory is to create the predictions and then the KB from a pre-trained model.

This repository is under construction and we're in the process of adding support for more datasets.


## Table of Contents
- [Dependencies](#dependencies)
- [Dataset Formatting](#dataset_formatting)


## Dependencies
Please refer to the dependencies in the main directory to run for curating the predictions.

## Dataset Formatting
In order to find the prediction over any dataset, the dataset should be in the format of DyGIE model input in which out models are trained. 
The common parameters that are needed includes:

- "senetences" : This parameter includes a list of sentences, each holding a list of tokens that combined they represent the sentence.
- "section" : This parameter shows which part of the text the data is from. All the data we tried on are from abstract but this value can represent any part of the publication.
- "dataset": This parameter should be set only for running the granular model and not the binary model. Please set this value to "covid-event" for running the pre-trained granular model. 

## CORD19 dataset preperation
The knowledge based introduced in the paper is from  the [CORD19 courpus](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge)
To convert the data from CORD19 to DyGIE you can use the script provided in this directory as ##TODO

 
