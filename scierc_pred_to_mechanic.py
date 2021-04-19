import argparse
import json
from typing import Any, Dict
import sys
from dygie_visualize_util import Dataset
import pathlib
from pathlib import Path
"""
Usage:
python scierc_pred_to_cofie.py --prediction_path ./predictions/cofie
"""
MECHANISM_KEY_MAP = ["USED-FOR"]
EFFECT_KEY_MAP = []
IGNORE_KEY_MAP = ['PART-OF','HYPONYM-OF','CONJUNCTION','FEATURE-OF','COMPARE','EVALUATE-FOR']

def get_doc_key_info(ds):
  doc_info_conf_iter = {}
  for doc in ds:
    doc_key = doc._doc_key
    for sent in doc:
      sent_text = " ".join(sent.text)
      for rel in sent.relations:
        arg0 = " ".join(rel.pair[0].text)
        arg1 = " ".join(rel.pair[1].text)
        data_key = (doc_key, sent_text, arg0, arg1, rel.label)
        doc_info_conf_iter[data_key] = rel.score
  return doc_info_conf_iter


def prediction_to_tsv(ds, output_file_name):  
  doc_info = get_doc_key_info(ds)
  output_file = open(output_file_name, "w")
  for key in doc_info:
    if key[4] in MECHANISM_KEY_MAP:
      conf0 = str(doc_info[key])
      output_file.write(key[0] + '\t' + key[1] + '\t' + key[2] + '\t' + key[3] + '\tMECHANISM\t' + conf0 + '\n')
    elif key[4] in MECHANISM_KEY_MAP:
      conf0 = str(doc_info[key])
      output_file.write(key[0] + '\t' + key[1] + '\t' + key[2] + '\t' + key[3] + '\tEFFECT\t' + conf0 + '\n')
    elif key[4] in IGNORE_KEY_MAP:
      continue
    else:
      conf0 = str(doc_info[key])
      output_file.write(key[0] + '\t' + key[1] + '\t' + key[2] + '\t' + key[3] + '\t' + str(key[4]) + '\t' + conf0 + '\n')



if __name__ == '__main__':

    parser = argparse.ArgumentParser() 

    parser.add_argument('--prediction_path',
                        type=str,
                        help='path to the directory containing dygie++ predictions in json fromat',
                        required=True)

    args = parser.parse_args()
    pred_path = pathlib.Path(args.prediction_path) / 'pred.json'

    ds = Dataset(pred_path)
    prediction_to_tsv(ds, pathlib.Path(pred_dir) / "pred.tsv")
