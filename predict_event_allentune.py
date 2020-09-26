# Predict, then uncollate
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
from dygie.data.dataset_readers import document
import pandas as pd
from decode import decode
"""
Usage:
python predict_event_allentune.py --serial_dir ./models/cofie-t --data_dir ./data/cofie-t/ --pred_dir ./predictions/cofie-t 
python predict_event_allentune.py --serial_dir ./models/cofie-t --data_dir ./data/cofie-t/ --pred_dir ./predictions/cofie-t --test_data --test_index 17
"""


def stringify(xs):
    return " ".join(xs)

def format_predicted_events(sent, doc_key=""):
    res = []
    for event in sent.predicted_events:
        if len(event.arguments) < 2:
          continue
        arg0 = event.arguments[0]
        arg1 = event.arguments[1]

        entry = {"doc_key": sent.metadata["_orig_doc_key"],
                 "sentence": stringify(sent.text),
                 "arg0": stringify(arg0.span.text),
                 "trigger": event.trigger.token.text,
                 "arg1": stringify(arg1.span.text),
                 "arg0_logit": arg0.raw_score,
                 "trigger_logit": event.trigger.raw_score,
                 "arg1_logit": arg1.raw_score,
                 "arg0_softmax": arg0.softmax_score,
                 "trigger_softmax": event.trigger.softmax_score,
                 "arg1_softmax": arg1.softmax_score}
        res.append(entry)
    return res


def format_dataset(dataset):
    predicted_events = []

    for doc in dataset:

        # import pdb; pdb.set_trace()
        for sent in doc:
          
            predicted = format_predicted_events(sent)
            predicted_events.extend(predicted)

    predicted_events = pd.DataFrame(predicted_events)

    return predicted_events

def load_jsonl(fname):
    return [json.loads(x) for x in open(fname)]


def save_jsonl(xs, fname):
    with open(fname, "w") as f:
        for x in xs:
            print(json.dumps(x), file=f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser() 

    parser.add_argument('--serial_dir',
                        type=str,
                        help="path to the saved trained model",
                        default="./models/events/")

    parser.add_argument('--data_dir', 
                        type=str,
                        help="path to the directory containing the test and dev data files",
                        default="data/processed/collated/")

    parser.add_argument('--pred_dir',
                          type=str,
                          help="path to the directory to save the prediction files",
                          default="./predictions/events/")

    parser.add_argument('--test_data',
                        action='store_true',
                        help="set this flag for script to predict over the test set. If set you should indicate the best performing dev index")
    
    parser.add_argument('--test_index',
                        type=int,
                        default=0,
                        help="best performing model on the development set")

    args = parser.parse_args()

    serial_dir = pathlib.Path(args.serial_dir)  
    pred_dir = pathlib.Path(args.pred_dir)
    test_dir = pathlib.Path(args.data_dir)
    if args.test_data:
      test_dir = pathlib.Path(test_dir) /'test.json'
    else:
      test_dir = pathlib.Path(test_dir) /'dev.json'
   

    for file in os.listdir(str(serial_dir)):
      trail_strat_str = "run_"
      if args.test_data:
        trail_strat_str = trail_strat_str + str(args.test_index)

      if file.startswith(trail_strat_str):
        run_serial_dir = serial_dir / file / "trial"
        uncollate_pred_dir = pred_dir / file
        uncollate_pred_dir.mkdir(parents=True, exist_ok=True)
        
        
        uncollated_pred_path = pathlib.Path(uncollate_pred_dir) / "pred.json"
        uncollated_pred_path_decode = pathlib.Path(uncollate_pred_dir) / "decode.json"
        uncollated_pred_path_tsv = pathlib.Path(uncollate_pred_dir) / "pred.tsv"


        allennlp_command = [
                  "allennlp",
                  "predict",
                  str(run_serial_dir),
                  str(test_dir),
                  "--predictor dygie",
                  "--include-package dygie",
                  "--use-dataset-reader",
                  "--output-file",
                  str(uncollated_pred_path),
                  "--cuda-device",
                  "0"
          ]
        try:
          subprocess.run(" ".join(allennlp_command), shell=True, check=True)
          
          in_data = load_jsonl(str(uncollated_pred_path))
          out_data = decode(in_data)
          save_jsonl(out_data, str(uncollated_pred_path_decode))
          dataset = document.Dataset.from_jsonl(str(uncollated_pred_path_decode))
          pred = format_dataset(dataset)
          pred.to_csv(str(uncollated_pred_path_tsv), sep="\t", float_format="%0.4f", index=False)
        except:
           os.rmdir(str(uncollate_pred_dir))
           continue
