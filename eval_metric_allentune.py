import argparse
import json
import os   
import shutil
import subprocess
from typing import Any, Dict
import sys
import pandas as pd
from eval_utils import depparse_base, allpairs_base, get_openie_predictor,get_srl_predictor,allenlp_base_relations, ie_eval
import pathlib
from pathlib import Path
import pandas as pd
"""
Usage:
python eval_metric_allentune.py --pred_path ./predictions/cofie/ --gold_path ./cofie-gold
python eval_metric_allentune.py --pred_path ./predictions/cofie/ --gold_path ./gold-gold/ --test_data --test_index 14
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
    gold_path = pathlib.Path(args.gold_path) / 'test-gold.tsv'
    pred_dir = pathlib.Path(args.pred_path) 
    coref = None        

    GOLD_PATH = pathlib.Path(gold_path)
    golddf = pd.read_csv(GOLD_PATH, sep="\t",header=None, names=["id","text","arg0","arg1","rel","y"])
    golddf = golddf[golddf["y"]=="accept"]
    best_run_index = 0
    best_run_score = 0

    for file in os.listdir(str(pred_dir)):
      trail_strat_str = "run_"
      if args.test_data:
         trail_strat_str = trail_strat_str + str(args.test_index)
      
      if file.startswith(trail_strat_str):
        run_pred_dir = pred_dir / file /"pred.tsv"
        PREDS_PATH = pathlib.Path(run_pred_dir)

        #read predictions, place in dictionary
        run_index = file[4:file.index('_', 5)]
        prediction_dict = {}
        try:
            predf = pd.read_csv(PREDS_PATH, sep="\t",names=["id","text","arg0","arg1","rel","conf"])
        except:
            continue
        if len(predf) > 1000:   #happpens in cases where model is not trained properly
            continue

        #check prediction label mapping matches the loaded gold file
        print(predf["rel"].unique())
        print(golddf["rel"].unique())
        prediction_dict['covid_run_' + run_index] = predf[["id","arg0","arg1","rel","conf"]]
        
        res_list = []

        for k,v in prediction_dict.items():
            trial_score = 0

            print(k)
        
            print ("****")
            if not len(v):
                print(k," -- NO PREDICTIONS -- ")
                continue
            #only try non-collapsed labels for relations that have it (i.e. ours and gold)
            if "rel" not in v.columns:
                collapse_opt = [True]
            else:
                collapse_opt = [False,True]
            for match_metric in ["exact","substring"]:
                for collapse in collapse_opt:

                    corr_pred, precision,recall, F1 = ie_eval(v,golddf,collapse = collapse,coref=coref, match_metric=match_metric,jaccard_thresh=0.5,consider_reverse=False, transivity=False)
                    res = [k, precision, recall, F1, collapse, match_metric, 0.5]
                    trial_score += F1
                    res_list.append(res)
                    print('model: {0} collapsed: {1} metric: {2} precision:{3} recall {4} f1: {5}'.format(k, collapse, match_metric, precision,recall, F1))
                    if match_metric == "jaccard":
                        corr_pred, precision,recall, F1 = ie_eval(v,golddf,collapse = collapse,coref=coref, match_metric=match_metric,jaccard_thresh=0.4,consider_reverse=False, transivity=False)
                        res = [k, precision, recall, F1, collapse, match_metric, 0.4]
                        res_list.append(res)
                        print('model: {0} collapsed: {1} metric: {2} precision:{3} recall {4} f1: {5}'.format(k, collapse, match_metric, precision,recall, F1))
                        corr_pred, precision,recall, F1 = ie_eval(v,golddf,collapse = collapse,coref=coref, match_metric=match_metric,jaccard_thresh=0.3,consider_reverse=False, transivity=False)

            if trial_score > best_run_score:
                best_run_score = trial_score
                best_run_index = k
    if args.test_data == False:
        print("best run is " + str(best_run_index))

