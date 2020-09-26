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
python predict.py --data_dir data/cofie --device 0,1,2,3 --serial_dir models/cofie/ 
python predict.py --data_dir data/cofie --device 0,1,2,3 --serial_dir models/cofie/  --pred_dir predictions/cofie
"""



def get_doc_key_info(ds):
  doc_info_conf_iter = {}
  for doc in ds:
    doc_key = doc._doc_key
    for sent in doc:
      sent_text = " ".join(sent.text)
      for rel in sent.relations:
        arg0 = " ".join(rel.pair[0].text).replace("\"", "")
        arg1 = " ".join(rel.pair[1].text).replace("\"", "")
        data_key = (doc_key, sent_text, arg0, arg1, rel.label)
        # print((doc_key, sent_text, arg0, arg1, rel.label))
        # import pdb;pdb.set_trace()
        doc_info_conf_iter[data_key] = rel.score
  return doc_info_conf_iter


def prediction_to_tsv(ds, output_file_name):  
  doc_info = get_doc_key_info(ds)
  print(len(doc_info))
  output_file = open(output_file_name, "w")
  for key in doc_info:
    conf0 = str(doc_info[key])
    output_file.write(key[0] + '\t' + key[1] + '\t' + key[2] + '\t' + key[3] + '\t' + str(key[4]) + '\t' + conf0 + '\n')


if __name__ == '__main__':

    parser = argparse.ArgumentParser() 

    parser.add_argument('--data_dir',
                        type=str, 
                        help="path to the directory containing the data file to make prediction on.",
                        required=True)

    parser.add_argument('--test_file',
                            type=str,
                            help="Please mention test filename in the data_dir if test filename is not test.json",
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

    parser.add_argument('--test_data',
                        action='store_true')
    
    parser.add_argument('--test_index',
                        type=int,
                        default=0)

    args = parser.parse_args()
    data_root = pathlib.Path(args.data_dir) 
    serial_dir = pathlib.Path(args.serial_dir)
    pred_dir = pathlib.Path(args.pred_dir)

    
    pred_dir.mkdir(parents=True, exist_ok=True)
    test_dir = data_root / args.test_file
    pred_path = pred_dir / args.pred_file



    if args.device:
        os.environ['CUDA_DEVICE'] = args.device
        os.environ['cuda_device'] = args.device

    for file in os.listdir(str(serial_dir)):
      print(file)
      trail_strat_str = "run_"
      if args.test_data:
        trail_strat_str = trail_strat_str + str(args.test_index)
      
      if file.startswith(trail_strat_str):
        run_serial_dir = serial_dir / file / "trial"
        run_pred_dir = pred_dir / file 

        run_pred_dir.mkdir(parents=True, exist_ok=True)
        
        
        pred_path = pathlib.Path(run_pred_dir) / "pred.json"


        allennlp_command = [
                  "allennlp",
                  "predict",
                  str(run_serial_dir),
                  str(test_dir),
                  "--predictor dygie",
                  "--include-package dygie",
                  "--use-dataset-reader",
                  "--output-file",
                  str(pred_path),
                  "--cuda-device",
                  args.device
          ]
        try:
          subprocess.run(" ".join(allennlp_command), shell=True, check=True)
          ds = Dataset(pred_path)
          prediction_to_tsv(ds, pathlib.Path(run_pred_dir) / "pred.tsv")
        except:
          pass


