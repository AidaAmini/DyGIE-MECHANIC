import pandas as pd
from pathlib import Path
import torch
from transformers import *
import string
import numpy as np
import logging
from txtai.embeddings import Embeddings
import argparse
from datetime import datetime
from tqdm import tqdm
import pickle


"""python create_kb_embeddings.py --sent_trans_path ../distilroberta-base-paraphrase-v1/0_Transformer/ --predictions_path ../complete_KB_coref.tsv --embeddings_index_path distil_embedding_index""" 
if __name__ == '__main__':

    parser = argparse.ArgumentParser() 

    parser.add_argument('--sent_trans_path',
                        type=Path,
                        help='path/name of sentence transformer model. could be created with training_multi-task.py. Should be something like ../biomed_roberta_base-2020-06-18_00-15-06/0_Transformer. ',
                        default="sentence-transformers/distilroberta-base-paraphrase-v1",
                        required=False)

    parser.add_argument('--predictions_path',
                        type=str,
                        required=True,
                        help="location of predicted relations tsv")

    parser.add_argument('--embeddings_index_path',
                        type=str,
                        default='embedding_index',
                        required=False,
                        help="where to save/load the embeddings index")

    parser.add_argument('--conf_thresh',
                        type=float,
                        default=0.9,
                        help="threshold for KB confidence filtering")

    args = parser.parse_args()


    
    sentence_transformer_path = Path(args.sent_trans_path)
    embeddings_index_path = Path(args.embeddings_index_path)
    predictions_path = Path(args.predictions_path)



    ### load predictions on CORD-19 abstracts to create KB
    kb = pd.read_csv(predictions_path,usecols=["doc_id","sentence","span1","span2","relation_tag","conf","span1_lemma","span2_lemma"],sep="\t")
    kb.dropna(inplace=True)
    #string cleanups 
    kb['norm_span1'] = kb['span1'].str.replace('[^\w\s]','').str.replace("\s\s+", " ").str.strip().str.replace('^(\d+\s ?)*|(^[0-9]+)', '').str.replace("^[0-9]+$","")
    kb['norm_span2'] = kb['span2'].str.replace('[^\w\s]','').str.replace("\s\s+", " ").str.strip().str.replace('^(\d+\s ?)*|(^[0-9]+)', '').str.replace("^[0-9]+$","")
    kb = kb[~((kb.norm_span1=="") | (kb.norm_span2==""))]
    badi = []
    for i in range(len(kb["conf"])):
        v = kb["conf"].iloc[i]
        try:
            float(v)
        except:
            badi.append(i)
    kb.drop(kb.index[badi],inplace=True)

    #Drop duplicates
    kb.drop_duplicates(["doc_id","span1_lemma","span2_lemma"],inplace=True)
    kb.drop_duplicates(["doc_id","norm_span1","norm_span2"],inplace=True)
    #Filter by confidence
    kb["conf"] = kb["conf"].astype(float)
    kb = kb[kb["conf"]>=args.conf_thresh]
    print(kb.shape)
    kb["doc_id"] = kb["doc_id"].str.split("_").str[0]
    kb[["doc_id","sentence","span1","span2","relation_tag","conf","norm_span1","norm_span2"]].to_csv("../fkb_filtered.csv",header=True,index=False)

    uniqueterms = pd.unique(kb[['norm_span1', 'norm_span2']].values.ravel('K'))
    print(uniqueterms.shape)

    #create index
    #uniqueindex is your list of terms
    embeddings = Embeddings({"method": "transformers", "path": sentence_transformer_path.__str__(),"quantize":True})
    embeddings.index([(uid, text, None) for uid, text in enumerate(uniqueterms)])
    embeddings.save("embedding_index")
   