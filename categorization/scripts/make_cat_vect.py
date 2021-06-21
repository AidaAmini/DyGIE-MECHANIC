import json
# from .global_vars import KMEANS
from add_namespace import embed_text, get_clusters, get_majority
from tf_idf import scrub
from global_vars import Globals
from joblib import load
"""
Given list of words get n-grams from it
"""
def n_grams(_list, start, n):
    n_grams_list = []
    for gram in range(start, n + 1):
        for index in range(len(_list) - gram + 1):
            words = " ".join(_list[index:index + gram])
            n_grams_list.append(words)
    # check just in case list empty
    if len(n_grams_list) == 0:
        n_grams_list = [" ".join(_list)]
    return n_grams_list

# def get_doubles(_list):
#     doubles_list = []
#     if len(_list) < 2:
#         return []
#     for buffer in range(1, 3):
#     # buffer = 2
#         for index in range(buffer, len(_list) - buffer):
#             words = " ".join(_list[index - buffer:index + buffer + 1])
#             doubles_list.append(words)
#     if len(doubles_list) == 0:
#         doubles_list = [" ".join(_list)]
#     return doubles_list

# gets weighted average of ngrams
def get_cluster_total(gram_clusters):
    cluster_totals = {}

    for cluster in range(Globals.CLUSTER_SIZE):
        cluster_totals[cluster] = 0

    for ans in gram_clusters:
        cluster_totals[ans] += 1
    cluster_len = len(gram_clusters)

    for cluster in range(Globals.CLUSTER_SIZE):
        cluster_totals[cluster] /= cluster_len
    return cluster_totals

def main(subfilename):
    old_train = open("../../data/mechanic/coarse_categorized/" + subfilename, "r")
    # add dataset to each field
    final_clust_data = {}
    output_file = open(FOLDER_PATH + Globals.NAME + "_total_" + subfilename, "w")
    count = 0
    
    for i in range(Globals.CLUSTER_SIZE):
        final_clust_data[i] = 0

    for line in old_train:
        count += 1
        line_json = json.loads(line)
        sentence_amt = len(line_json["sentences"])

        total_kmeans_preds = []
        doc_key = line_json["doc_key"]
        for index in range(sentence_amt):
            sentence = line_json["sentences"][index]
            # ners = line_json["ner"][index]
            # dataset = line_json["dataset"]
            

            remaining_words = scrub(sentence)
            # remaining_words = sentence
            grams = n_grams(remaining_words, 2, 5)
            # clusters = get_clusters(remaining_words)
            # maj = get_majority(clusters)
            gram_clusters = get_clusters(grams)
            total_kmeans_preds += gram_clusters.tolist() 

        clust_total = get_cluster_total(total_kmeans_preds)
        output_file.write(doc_key + "\t" + str(clust_total) + "\n")
        for i in range(Globals.CLUSTER_SIZE):
            final_clust_data[i] += clust_total[i]

    for i in range(Globals.CLUSTER_SIZE):
        final_clust_data[i] /= count
    FILE.write(str(final_clust_data)+"\n")
    output_file.close()
    old_train.close()
FOLDER_PATH = "data/"
FILE = open(FOLDER_PATH + "total_metrics.txt", "w")

if __name__ == "__main__":
    for kmeans in [
        (load("/data/edan/categorization/nosymbol_cls_8_kmeans/revised_kmeans.joblib"), 8, "cls", "nosymbol_cls_8"),
        (load("/data/edan/categorization/nosymbol_revmean_10_kmeans/revised_kmeans.joblib"), 10, "mean", "nosymbol_revmean_10"),
        (load("/data/edan/categorization/nosymbol_cls_10_kmeans/revised_kmeans.joblib"), 10, "cls", "nosymbol_cls_10"),
        (load("/data/edan/categorization/symbol_mean_10_kmeans/training_kmeans.joblib"), 10, "simple_mean", "symbol_mean_10")
    ]:
        Globals.KMEANS = kmeans[0]
        Globals.CLUSTER_SIZE = kmeans[1]
        Globals.TYPE = kmeans[2]
        Globals.NAME = kmeans[3]
        for names in ["test.json", "train.json", "dev.json"]:
            main(names)

    # .mean(1).detach().numpy()[0]