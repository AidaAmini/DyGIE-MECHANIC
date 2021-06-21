import torch
from transformers import BertTokenizer, BertModel
from sklearn.cluster import KMeans
from joblib import dump, load
from tf_idf import scrub, symbol_scrub
import numpy as np
from add_namespace import grab_mean

import sys

print("starting run")
  # prediction path
# INPUT_SPAN_FILE_PATH = "/data/edan/for_edan/cofie_spans.txt"
model_version = 'allenai/scibert_scivocab_uncased'
do_lower_case = True
model = BertModel.from_pretrained(model_version)
model.to('cuda:1')
tokenizer = BertTokenizer.from_pretrained(model_version, do_lower_case=do_lower_case)

def embed_text(text, model):
    input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0)  # Batch size 1
    input_ids = input_ids.to('cuda:1')
    outputs = model(input_ids)
    last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
    return last_hidden_states 


def get_words(input_file):
  word_list = []
  word_em_list = []
  count = 0
  for line in input_file:
    if count% 10000 == 0:
      print(count)
    # if count == 100000:
    #   break
    count += 1
    if line[:-1] not in word_list:
      words = line[:-1].split(" ")
      sentence = " ".join(symbol_scrub) # delete symbols
      word_list.append(sentence)
      mean = grab_mean(sentence)
      word_em_list.append(mean)
  return word_list, word_em_list

def main(INPUT_SPAN_FILE_PATH):
  input_file = open(INPUT_SPAN_FILE_PATH)


  word_list, word_em_list = get_words(input_file)

  kmeans = KMeans(init="random", n_clusters=10, n_init=20, max_iter=300, random_state=42)
  kmeans.fit(word_em_list)

  # create training clusters
  output_file = open("k_mean_clusters_training.tsv", "w")
  for i in range(len(kmeans.labels_)):
    output_file.write(word_list[i]+ "\t" + str(kmeans.labels_[i]) + "\n")

  dump(kmeans, "revised_kmeans.joblib")
  print("words dumped")

if __name__ == "__main__":
  INPUT_SPAN_FILE_PATH = "/data/edan/for_edan/training_spans.txt"
  main(INPUT_SPAN_FILE_PATH)