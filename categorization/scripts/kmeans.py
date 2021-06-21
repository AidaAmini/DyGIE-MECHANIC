import torch
from transformers import BertTokenizer, BertModel
from sklearn.cluster import KMeans
from joblib import dump, load

import sys

print("starting run")
  # prediction path
# INPUT_SPAN_FILE_PATH = "/data/edan/for_edan/cofie_spans.txt"
INPUT_SPAN_FILE_PATH = "/data/edan/for_edan/training_spans.txt"

def embed_text(text, model):
    input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0)  # Batch size 1
    outputs = model(input_ids)
    last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
    return last_hidden_states 

model_version = 'allenai/scibert_scivocab_uncased'
do_lower_case = True
model = BertModel.from_pretrained(model_version)
tokenizer = BertTokenizer.from_pretrained(model_version, do_lower_case=do_lower_case)
input_file = open(INPUT_SPAN_FILE_PATH)

def get_words(input_file, model):
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
      sentence = " ".join([x for x in words if len(x) != 1 or x.isalnum()]) # delete symbols
      word_list.append(sentence)
      word_em_list.append(embed_text(word_list[-1], model)[0, 0].detach().cpu().numpy())
  return word_list, word_em_list

word_list, word_em_list = get_words(input_file, model)

dump(word_list, "revised_wordlist.joblib")
dump(word_em_list, "revised_wordemlist.joblib")

print("words dumped")
