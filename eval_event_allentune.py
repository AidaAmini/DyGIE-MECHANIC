import argparse
import json
import os   
import shutil
import subprocess
from typing import Any, Dict
import sys
import pandas as pd
from eval_utils import read_coref_file, depparse_base, allpairs_base, get_openie_predictor,get_srl_predictor,allenlp_base_relations, ie_eval_event, ie_span_eval, ie_errors
import pathlib
from pathlib import Path
import pandas as pd
from tabulate import tabulate
"""
Usage:
python eval_event_allentune.py --pred_path ./predictions/cofie-t/ --gold_path ./cofie-t-gold/
python eval_event_allentune.py --pred_path ./predictions/cofie-t/ --gold_path ./cofie-t-gold/ --test_data --test_index 17
"""

if __name__ == '__main__':

    parser = argparse.ArgumentParser() 
    parser.add_argument('--pred_path',
                        type=Path,
                        help='dataset folder, contains the predicted files',
                        default="",
                        required=True)

    parser.add_argument('--gold_path',
                        type=Path,
                        help='folder, contains the gold data',
                        default="",
                        required=True)

    parser.add_argument('--test_data',
                        action='store_true')
    
    parser.add_argument('--test_index',
                        type=int,
                        default=0)


    args = parser.parse_args()
    if args.test_data:
        gold_path = pathlib.Path(args.gold_path) / 'test-gold.tsv'
    else:
        gold_path = pathlib.Path(args.gold_path) / 'dev-gold.tsv'

    pred_dir = pathlib.Path(args.pred_path) 
    coref = None      

    
    GOLD_PATH = pathlib.Path(gold_path)
    PREDS_PATH = pathlib.Path(pred_dir)
    golddf = pd.read_csv(GOLD_PATH, sep="\t",header=None, names=["id","text","arg0","trigger","arg1"])

    best_run_index = 0
    best_run_score = 0

    prediction_dict = {}
    for file in os.listdir(str(pred_dir)):

      trail_strat_str = "run_"
      if args.test_data:
         trail_strat_str = trail_strat_str + str(args.test_index)
      
      if file.startswith(trail_strat_str):


        run_pred_dir = pred_dir / file /"pred.tsv"
        PREDS_PATH = pathlib.Path(run_pred_dir)
        
        #read predictions, place in dictionary
        run_index = file[4:file.index('_', 5)]

        try:
            predf = pd.read_csv(PREDS_PATH, sep="\t",names=["id","text","arg0","trigger","arg1","arg0_logit","trigger_logit","arg1_logit","arg0_softmax","trigger_softmax","arg1_softmax"])
        except:
            continue
        if len(predf) > 1000:
            continue

        prediction_dict[str("events") + '_run_' + run_index] = predf[["id","arg0","trigger","arg1"]]
    
    #get results
    res_list = []
    res_latex_list = []
    for k,v in prediction_dict.items():
        print(k)
        trial_score = 0.0
        print ("****")
        if not len(v):
            print(k," -- NO PREDICTIONS -- ")
            continue
        #only try non-collapsed labels for relations that have it (i.e. ours and gold)
        collapse_opt = [False,True]
        for match_metric in ["substring","exact"]:

            for consider_reverse in [False]:
                for collapse in collapse_opt:
                    th_opts = [1]
                    for th in th_opts:
                        corr_pred, precision,recall, F1 = ie_eval_event(v,golddf,coref=coref,collapse = collapse, match_metric=match_metric,jaccard_thresh=th,consider_reverse=consider_reverse,transivity=False)
                        trial_score += F1
                        res = [k, 100*round(precision,3), 100*round(recall,3), 100*round(F1,3), collapse, match_metric, th, consider_reverse]
                        # if collapse == True and consider_reverse == True:
                        res_latex = [k, match_metric, 100*round(precision,3), 100*round(recall,3), 100*round(F1,3)]
                        res_latex_list.append(res_latex)
                        res_list.append(res)
        if trial_score > best_run_score:
            best_run_score = trial_score
            best_run_index = k
       
    print(tabulate(res_list, headers =["model","P","R","F1","collapse","match_mettric","threshold", "consider_reverse"]))
    print ("****")
    if args.test_data == False:
            print("best run is " + str(best_run_index))
   

