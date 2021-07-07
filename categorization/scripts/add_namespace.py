"""
Adds dataset to provided namespaces
"""
from joblib import load
from scipy.sparse import data
import os
import shutil
import json
import torch
import random
from global_vars import Globals
# MODEL, TOKENIZER, KMEANS, TYPE
from tf_idf import scrub
import numpy as np
import copy
import ast


# load k means clusterer
PATH = "coarse/test/"

def init():
    try:
        shutil.rmtree(PATH)
    except OSError as e:
        print("No old version of", os.getcwd() + "/" + PATH, "to delete")
    os.mkdir(PATH)

def get_spans(sentence, ners):
    spans = []
    for ner in ners:
        span_start = ner[0]
        span_end = ner[1]
        spans.append(" ".join(sentence[span_start:span_end + 1]))
    return spans

"""

"""
def embed_text(text):
    input_ids = torch.tensor(Globals.TOKENIZER.encode(text)).unsqueeze(0)  # Batch size 1
    outputs = Globals.MODEL(input_ids)
    last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
    return last_hidden_states 

def grab_mean(span):
  # get split words
  words = Globals.TOKENIZER.tokenize(span)
  non_stop_words = set(scrub(words))
  embeddings = embed_text(span)
  final_embeddings = []
  for index, word in enumerate(words):
    if word in non_stop_words:
      final_embeddings.append(embeddings[0, index + 1].detach().cpu().numpy()) 
  if (len(final_embeddings) == 0):
    final_embeddings.append(embeddings[0, 0].detach().cpu().numpy())
  return np.asarray(final_embeddings).mean(0)

"""
Get CLS
"""
def grab_cls(span):
    return embed_text(span).detach()[0, 0].cpu().numpy()

def grab_simple_mean(span):
    return embed_text(span).detach().mean(1).cpu().numpy().flatten()

"""
Given KMEANS clusters and spans generate predictions
"""
def get_clusters(spans):

    span_emb_list = []

    if Globals.TYPE=="mean":
        for span in spans:
            span_emb_list.append(grab_mean(span))
    elif Globals.TYPE=="cls":
        for span in spans:
            span_emb_list.append(grab_cls(span))
    elif Globals.TYPE=="simple_mean":
        for span in spans:
            span_emb_list.append(grab_simple_mean(span))
    else:
        raise Exception("Unrecognized type: " + str(Globals.TYPE))

    preds = Globals.KMEANS.predict(span_emb_list)
    return preds

"""
get majority from KMEANS predictions
"""
def get_majority(preds):
    if (len(preds) == 0):
        raise Exception("No predictions for a sentence")

    # create dict of occurances of cluster
    clusters = {}
    for pred in preds:
        if pred not in clusters:
            clusters[pred] = 1
        else:
            clusters[pred] += 1
    
    # grab only the max clusters
    max = 0
    max_list = []
    for key in clusters.keys():
        val = clusters[key]
        if (val > max):
            max_list = [key]
            max = val
        elif (val == max):
            max_list.append(key)

    if (len(max_list) > 1):
        print("List of size greater than 1, choosing a cluster at random from", max_list)
    return random.choice(max_list)
        

        

def print_for_docs(_dict, subfilename):
    stat_file = open(PATH + Globals.NAME + "_stats_" + subfilename, "w")
    stat_file.write("\t".join(_dict.keys()) + "\n")
    stat_file.write("\t".join(str(x) for x in _dict.values()) + "\n")
    stat_file.close()

"""
get ners cooresponding to indexes
"""
def get_ners(indexes, ners):
    new_ners = [[] for row in ners]
    for index in indexes:
        offset = 0
        for row_index, row in enumerate(ners):
            row_len = len(row)
            if offset + row_len > index:
                new_ners[row_index].append(row[index - offset])
                continue
            offset += row_len
    return new_ners
    
def add_dataset(subfilename, test=False):
    new_train = open(PATH + Globals.NAME + "_" + subfilename, "w") 
    old_train = open("../../data/mechanic/coarse_uncategorized/" + subfilename, "r")
    # add dataset to each field
    clust_data = {}

    for count, line in enumerate(old_train):
        line_json = json.loads(line)

        if test==True:
            doc_key = line_json["doc_key"]
            top_n = get_top_n(Globals.MAP[doc_key], 3)

            for dataset in top_n:
                line_json["dataset"] = dataset
                new_train.write(json.dumps(line_json) + "\n")

                dataset = str(dataset)
                if (dataset not in clust_data):
                    clust_data[dataset] = 1
                else:
                    clust_data[dataset] += 1
        else:
            sentence_amt = len(line_json["sentences"])
            total_KMEANS_preds = []
            for index in range(sentence_amt):
                sentence = line_json["sentences"][index]
                ners = line_json["ner"][index]

                spans = get_spans(sentence, ners)
                KMEANS_preds = get_clusters(spans)
                total_KMEANS_preds = total_KMEANS_preds + KMEANS_preds.tolist()

            # seperate predictions into different lines
            pred_map = {}
            for i, pred in enumerate(total_KMEANS_preds):
                if pred not in pred_map:
                    pred_map[pred] = []
                pred_map[pred].append(i)

            # write these lines to file
            for dataset in pred_map.keys():
                copy_json = copy.deepcopy(line_json)
                copy_json["dataset"] = dataset

                copy_ners = get_ners(pred_map[dataset], line_json["ner"])
                copy_json["ner"] = copy_ners

                # Only change names in training not in testing
                copy_json["doc_key"] += "___" + str(dataset)
            
                new_train.write(json.dumps(copy_json) + "\n")
                
                dataset = str(dataset)
                if (dataset not in clust_data):
                    clust_data[dataset] = 1
                else:
                    clust_data[dataset] += 1


        # dataset = str(get_majority(total_KMEANS_preds))
        # line_json["dataset"] = dataset
        # new_train.write(json.dumps(line_json) + "\n")

        
    print_for_docs(clust_data, subfilename)
    
    new_train.close()
    old_train.close()

def get_map(file_name):
    map = {}
    f = open(file_name, "r")
    for line in f:
        _tuple = line.split("\t")
        map[_tuple[0]] = ast.literal_eval(_tuple[1][:-1])
    f.close()
    return map

# slow function, only useful since list is very small
def get_top_n(_list, n):
    _list = copy.copy(_list)
    assert len(_list) >= n
    top_indexes = []
    for i_n in range(n):
        max_val = -1
        max_index = -1
        for index in range(len(_list)):
            val = _list[index]
            if val > max_val:
                max_val = val
                max_index = index
        _list[max_index] = -1
        top_indexes.append(max_index)
    return top_indexes
    
    
        


if __name__ == "__main__":
    init()
    for kmeans in [
        (load("/data/edan/categorization/nosymbol_cls_8_kmeans/revised_kmeans.joblib"), 8, "cls", "nosymbol_cls_8", "data/nosymbol_cls_8_total_"),
        (load("/data/edan/categorization/nosymbol_revmean_10_kmeans/revised_kmeans.joblib"), 10, "mean", "nosymbol_revmean_10", "data/nosymbol_revmean_10_total_"),
        (load("/data/edan/categorization/nosymbol_cls_10_kmeans/revised_kmeans.joblib"), 10, "cls", "nosymbol_cls_10", "data/nosymbol_cls_10_total_"),
        (load("/data/edan/categorization/symbol_mean_10_kmeans/training_kmeans.joblib"), 10, "simple_mean", "symbol_mean_10", "data/symbol_mean_10_total_")
    ]:
        Globals.KMEANS = kmeans[0]
        Globals.CLUSTER_SIZE = kmeans[1]
        Globals.TYPE = kmeans[2]
        Globals.NAME = kmeans[3]
        
        subfiles = ["dev.json", "test.json"]
        for subfilename in subfiles:
            Globals.MAP = get_map(kmeans[4] + subfilename)
            add_dataset(subfilename, test=True)