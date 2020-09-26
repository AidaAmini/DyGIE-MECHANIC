import argparse
import json
import os   
import shutil
import subprocess
from typing import Any, Dict
import sys
import pandas as pd
from eval_utils import read_coref_file, depparse_base, allpairs_base, get_openie_predictor,get_srl_predictor,allenlp_base_relations, ie_eval, ie_span_eval, ie_errors
import pathlib
from pathlib import Path
import pandas as pd
from tabulate import tabulate

"""
Usage:
source activate covid_eval
python eval_metric.py --gold_dir ./cofie-gold/ --pred_dir ./predictions/cofie/
python eval_metric.py --gold_dir ./cofie-gold/ --pred_dir ./predictions/cofie/  --latex_print
"""

if __name__ == '__main__':

    parser = argparse.ArgumentParser() 
    parser.add_argument('--pred_dir',
                        type=str,
                        help='path to predictions',
                        required=True)
    parser.add_argument('--gold_dir',
                        type=str,
                        help='path to gold labels',
                        required=False)
    parser.add_argument('--stats_dir',
                        type=str,
                        help='path to save the eval metrics, default is stats/stats.tsv',
                        default="stats/stats.tsv",
                        required=False)

    parser.add_argument('--dev_mode',
                        action='store_true', 
                        help="if set true the prediction is measured with development gold set.")

    parser.add_argument('--open',
                        action='store_true',
                        help="if set to be true, the srl and openie baseline are also calculated.")

    parser.add_argument('--latex_print',
                        action='store_true', 
                        help="if set the latex format for tha paper is printed")

    args = parser.parse_args()
    gold_path = pathlib.Path(args.gold_dir)
    pred_path = pathlib.Path(args.pred_dir)
    stat_path = pathlib.Path(args.stats_dir)
    stat_path.mkdir(parents=True, exist_ok=True)


    coref = None
    if args.dev_mode:
        GOLD_PATH = gold_path / 'dev-gold.tsv'
    else:
        GOLD_PATH = gold_path / 'test-gold.tsv'

    PREDS_PATH = pred_path / 'pred.tsv'
    golddf = pd.read_csv(GOLD_PATH, sep="\t",header=None, names=["id","text","arg0","arg1","rel","y"])
    golddf = golddf[golddf["y"]=="accept"]
    #read predictions, place in dictionary

    prediction_dict = {}
    predf = pd.read_csv(PREDS_PATH, sep="\t",names=["id","text","arg0","arg1","rel","conf"])
    prediction_dict["covid_model"] = predf[["id","arg0","arg1","rel","conf"]]


    # get SRL relations and openIE relations, place in prediction_dict
    if args.open:
        predictor_ie = get_openie_predictor()
        predictor_srl = get_srl_predictor()
        use_collapse = False
        srl_relations = allenlp_base_relations(predictor_srl,golddf,filter_biosrl=False,collapse=use_collapse)
        srl_relations_fileter = allenlp_base_relations(predictor_srl,golddf,filter_biosrl=True,collapse=use_collapse)
        ie_relations = allenlp_base_relations(predictor_ie,golddf,filter_biosrl=False,collapse=use_collapse)
        ie_relations_filter = allenlp_base_relations(predictor_ie,golddf,filter_biosrl=True,collapse=use_collapse)
        if use_collapse:
            prediction_dict["srl"] = pd.DataFrame(srl_relations,columns=["id","arg0","arg1"])
            prediction_dict["srl-fl"] = pd.DataFrame(srl_relations_fileter,columns=["id","arg0","arg1"])
        else:
            prediction_dict["srl"] = pd.DataFrame(srl_relations,columns=["id","arg0","arg1","rel","conf"])
            prediction_dict["srl-fl"] = pd.DataFrame(srl_relations_fileter,columns=["id","arg0","arg1","rel","conf"])
        if use_collapse:
            prediction_dict["openie"] = pd.DataFrame(ie_relations,columns=["id","arg0","arg1"])
            prediction_dict["openie-fl"] = pd.DataFrame(ie_relations,columns=["id","arg0","arg1"])
        else:
            prediction_dict["openie"] = pd.DataFrame(ie_relations_filter,columns=["id","arg0","arg1","rel","conf"])
            prediction_dict["openie-fl"] = pd.DataFrame(ie_relations_filter,columns=["id","arg0","arg1","rel","conf"])
    
    #get results
    res_list = []
    res_latex_list = []
    res_latex_f1 = []
    res_span_list = []

    for k,v in prediction_dict.items():
        print ("****")
        print(k)
        if not len(v):
            print(k," -- NO PREDICTIONS -- ")
            continue

        collapse_opt = [False, True]
        latex_line = []
        name = k
        if "covid" in name:
            name = "covid"
        latex_line.append(name)
        for match_metric in ["exact", "rouge","substring"]: #ADDED LAST removed jaccard
            for consider_reverse in [False]:
                for reverse_on_effect in [True]:
                    for collapse in collapse_opt:
                        for transivity in [False]: #ADDED LAST  considering transivity 
                            th_opts = [1]
                            if match_metric == "rouge":
                                th_opts=[0.5]   #ADDED LAST chanding threshold for rouge
                            for th in th_opts:
                                p_at_k = []
                                k_th = [100, 150, 200, 50]
                                for topK in k_th:
                                    if "covid" not in k:
                                        p_at_k.append(0)
                                        continue
                                    _, p, _, _ = ie_eval(v,golddf,transivity=transivity,coref=coref,collapse = collapse, match_metric=match_metric,jaccard_thresh=th,topK=topK,consider_reverse=consider_reverse,reverse_on_effect=reverse_on_effect)
                                    p_at_k.append(p)
                                try:
                                    corr_pred, precision,recall, F1 = ie_eval(v,golddf,transivity=transivity,coref=coref,collapse = collapse, match_metric=match_metric,jaccard_thresh=th,consider_reverse=consider_reverse,reverse_on_effect=reverse_on_effect)
                                except:
                                        precision = 0
                                        recall = 0
                                        F1 = 0
                                        corr_pred= []
                                
                                span_corr_pred, span_precision,span_recall, span_F1 = ie_span_eval(v,golddf, match_metric=match_metric,jaccard_thresh=th)
                                ##writing
                                latex_line.append(100*round(F1,3))
                                if collapse == True:
                                    latex_line.append(100*round(span_F1,3))

                                res = [k, 100*round(precision,3), 100*round(recall,3), 100*round(F1,3), 100*round(p_at_k[0],3),100*round(p_at_k[1],3),100*round(p_at_k[2],3), 100*round(span_precision, 3), 100*round(span_recall, 3), 100*round(span_F1, 3), collapse, match_metric, th, consider_reverse]
                                
                                res_latex = [name, match_metric, th, collapse, 100*round(precision,3), 100*round(recall,3), 100*round(F1,3), 100*round(p_at_k[3],3),100*round(p_at_k[0],3), 100*round(span_precision, 3), 100*round(span_recall, 3), 100*round(span_F1, 3)]
                                res_latex_list.append(res_latex)
                                res_span = [k, span_precision, span_recall, span_F1, match_metric, th]
                                res_list.append(res)
                                
                                res_span_list.append(res_span)
        print(len(latex_line))
        res_latex_f1.append(latex_line)
       
    print(tabulate(res_list, headers =["model","P","R","F1","P@100","P@150","P@200","span_P","span_R","span_F1","collapse","match_mettric","threshold", "consider_reverse"]))
    print ("****")
    stats_df = pd.DataFrame(res_list,columns =["model","P","R","F1","P@100","P@150","P@200","span_P","span_R","span_F1","collapse","match_mettric","threshold", "consider_reverse"])
    stats_path = stat_path / 'stats.tsv'
    stats_df.to_csv(stats_path,header=True,index=False, sep="\t")
    if args.latex_print:
        stats_df_latex = pd.DataFrame(res_latex_list,columns =["model","metric", "th","collapse","P","R","F1","P@50","P@100","span_P","span_R","span_F1"]).set_index('model')
        stats_df_latex_f1 = pd.DataFrame(res_latex_f1,columns =["model","F1","F1-a","F1-ner","F1","F1-a","F1-ner","F1","F1-a","F1-ner"]).set_index('model')
        print(str(stats_df_latex_f1.to_latex()))
        print(str(stats_df_latex.to_latex()))
    
