"""
This file generates a kmeans cluster based on the provided input path and saves it to an output file
"""

import torch
from transformers import BertTokenizer, BertModel
from sklearn.cluster import KMeans

INPUT_SPAN_FILE_PATH = "/data/aida/covid_related/covid_clean/dygiepp_after_aaai/cofie_spans.txt"
INPUT_SPAN_FILE_PATH = "training_spans.txt"

def embed_text(text, model):
    input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0)  # Batch size 1
    outputs = model(input_ids)
    last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
    return last_hidden_states 

model_version = 'scibert_scivocab_uncased'
do_lower_case = True
model = BertModel.from_pretrained(model_version)
tokenizer = BertTokenizer.from_pretrained(model_version, do_lower_case=do_lower_case)
input_file = open(INPUT_SPAN_FILE_PATH)

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
    word_list.append(line[:-1])
    word_em_list.append(embed_text(word_list[-1], model).mean(1).detach().numpy()[0])


kmeans = KMeans(init="random", n_clusters=10, n_init=20, max_iter=300, random_state=42)
kmeans.fit(word_em_list)
output_file = open("k_mean_clusters_training.tsv", "w")
for i in range(len(kmeans.labels_)):
  output_file.write(word_list[i]+ "\t" + str(kmeans.labels_[i]) + "\n")

