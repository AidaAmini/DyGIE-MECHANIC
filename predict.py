import argparse
import json
import os
import shutil
import subprocess
from typing import Any, Dict
import sys
from dygie_visualize_util import Dataset
import pathlib
from pathlib import Path

"""
Usage
python predict.py --data_path data/cofie --device 0,1,2,3 --serial_dir models/cofie/ 
python predict.py --data_path data/cofie --device 0,1,2,3 --serial_dir models/cofie/  --pred_dir predictions/cofie

"""




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
  print("writing tsv formatted file : " + str(output_file_name))
  output_file = open(output_file_name, "w")
  for key in doc_info:
    conf0 = str(doc_info[key])
    output_file.write(key[0] + '\t' + key[1] + '\t' + key[2] + '\t' + key[3] + '\t' + str(key[4]) + '\t' + conf0 + '\n')


if __name__ == '__main__':

    parser = argparse.ArgumentParser() 



    parser.add_argument('--data_path',
                        type=str, 
                        help="path to the directory containing the data file to make prediction on.",
                        required=True)

    parser.add_argument('--test_file',
                            type=str,
                            help="Please mention test filename in the data_path if test filename is not test.json",
                            required=False,
                            default="test.json")

    parser.add_argument('--device',
                        type=str,
                        default='0',
                        required=False,
                        help="cuda devices comma seperated")

    parser.add_argument('--serial_dir',
                            type=str,
                            help="Path to the directory to save the model. default is ./models/",
                            required=True,
                            default="./models/")

    parser.add_argument('--pred_dir',
                            type=str,
                            help="Path to the directory to save the prediction. default is ./predictions/",
                            required=False,
                            default="./predictions/")

    parser.add_argument('--pred_file',
                            type=str,
                            help="Please mention prediction filename(including json extention) in the pred_dir if prediction filename should not be pred.json / pred.tsv",
                            required=False,
                            default="pred.json")

    args = parser.parse_args()

    data_root = pathlib.Path(args.data_path) 
    serial_dir = pathlib.Path(args.serial_dir)
    pred_dir = pathlib.Path(args.pred_dir)

    
    pred_dir.mkdir(parents=True, exist_ok=True)
    test_dir = data_root / args.test_file
    pred_path = pred_dir / args.pred_file

    allennlp_command = [
              "allennlp",
              "predict",
              str(serial_dir),
              str(test_dir),
              "--predictor dygie",
              "--include-package dygie",
              "--use-dataset-reader",
              "--output-file",
              str(pred_path),
              "--cuda-device",
              args.device
      ]
    print(" ".join(allennlp_command))
    subprocess.run(" ".join(allennlp_command), shell=True, check=True)
    ds = Dataset(pred_path)
    pred_name = args.pred_file.split('.')[0] + '.tsv'
    prediction_to_tsv(ds, pathlib.Path(pred_dir) / pred_name)
    
