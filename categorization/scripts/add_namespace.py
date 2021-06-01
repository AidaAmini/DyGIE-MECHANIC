from joblib import load
from scipy.sparse import data
from sklearn.cluster import KMeans
import os
import shutil
import json
from transformers import BertTokenizer, BertModel
import torch
import random

# load k means clusterer
path = "coarse/"

def init():
    try:
        shutil.rmtree(path)
    except OSError as e:
        print("No old version of", os.getcwd() + "/" + path, "to delete")
    os.mkdir(path)
    kmeans = load("/data/edan/categorization/training_kmeans.joblib")
    return kmeans

def get_spans(sentence, ners):
    spans = []
    for ner in ners:
        span_start = ner[0]
        span_end = ner[1]
        spans.append(" ".join(sentence[span_start:span_end + 1]))
    return spans


def get_clusters(kmeans, spans):
    def embed_text(text, model, tokenizer):
        input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
        return last_hidden_states 

    model_version = 'allenai/scibert_scivocab_uncased'
    do_lower_case = True
    model = BertModel.from_pretrained(model_version)
    tokenizer = BertTokenizer.from_pretrained(model_version, do_lower_case=do_lower_case)

    span_emb_list = []
    for span in spans:
        span_emb_list.append(embed_text(span, model, tokenizer).mean(1).detach().numpy()[0])

    preds = kmeans.predict(span_emb_list)
    return preds

"""
get majority from kmeans predictions
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
    stat_file = open(path + "stats_" + subfilename, "w")
    stat_file.write("\t".join(_dict.keys()) + "\n")
    stat_file.write("\t".join(str(x) for x in _dict.values()) + "\n")
    stat_file.close()

    
def add_dataset(subfilename, kmeans):
    new_train = open(path + subfilename, "w") 
    old_train = open("../../data/mechanic/coarse/" + subfilename, "r")
    # add dataset to each field
    clust_data = {}

    for count, line in enumerate(old_train):
        line_json = json.loads(line)
        sentence_amt = len(line_json["sentences"])

        total_kmeans_preds = []
        for index in range(sentence_amt):
            sentence = line_json["sentences"][index]
            ners = line_json["ner"][index]

            spans = get_spans(sentence, ners)
            kmeans_preds = get_clusters(kmeans, spans)
            total_kmeans_preds = total_kmeans_preds + kmeans_preds.tolist()

        dataset = str(get_majority(total_kmeans_preds))
        line_json["dataset"] = dataset
        new_train.write(json.dumps(line_json) + "\n")

        if (dataset not in clust_data):
            clust_data[dataset] = 1
        else:
            clust_data[dataset] += 1
        
    print_for_docs(clust_data, subfilename)
    
    new_train.close()
    old_train.close()

if __name__ == "__main__":
    kmeans = init()
    subfiles = ["train.json", "test.json", "dev.json"]
    for subfilename in subfiles:
        add_dataset(subfilename, kmeans)