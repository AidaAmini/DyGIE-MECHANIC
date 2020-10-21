import pandas as pd
from pathlib import Path
import torch
from transformers import *
import string
import numpy as np
import logging
from txtai.embeddings import Embeddings

from datetime import datetime
from tqdm import tqdm
import pickle

from task_queries import query_ai_uses, query_scifact
from KG_search_utils import get_similar_spans,get_sim,get_retrieved_rels_table, get_results


if __name__ == '__main__':

    parser = argparse.ArgumentParser() 

    parser.add_argument('--sent_trans_path',
                        type=Path,
                        help='path/name of sentence transformer model. could be created with training_multi-task.py. Should be something like ../biomed_roberta_base-2020-06-18_00-15-06/0_Transformer. ',
                        default="sentence-transformers/'distilroberta-base-paraphrase-v1",
                        required=True)

    parser.add_argument('--predictions_path',
                        type=str,
                        required=True,
                        help="location of predicted relations tsv")

    parser.add_argument('--create_embed_index',
                        action='store_true',
                        help = "if specified, will create new FAISS embeding index; otherwise, will load existing")

    parser.add_argument('--embeddings_index_path',
                        type=str,
                        default='embedding_index',
                        required=False,
                        help="where to save/load the embeddings index")

    parser.add_argument('--conf_thresh',
                        type=float,
                        default=0.9,
                        help="threshold for KB confidence filtering")

    parser.add_argument('--task',
                        type=str,
                        required=False,
                        choices =["ai","scifact"],
                        help="which evaluation task")

            

    args = parser.parse_args()


    
    sentence_transformer_path = Path(args.sent_trans_path)
    embeddings_index_path = Path(args.embeddings_index_path)
    predictions_path = Path(args.predictions_path)

    if args.create_embed_index:
        #create index
        #uniqueindex is your list of terms
        embeddings = Embeddings({"method": "transformers", "path": sentence_transformer_path.__str__(),"quantize":True})
        embeddings.index([(uid, text, None) for uid, text in enumerate(uniqueterms)])
        embeddings.save("embedding_index")
    else:
        #load index 
        embeddings = Embeddings()
        #hack to port an embedding_index created on another machine with other dir structure
        with open("%s/config" % "embedding_index", "rb") as handle:
                config = pickle.load(handle)
        config["path"] = sentence_transformer_path.__str__()
        with open("%s/config" % "embedding_index", "wb") as handle:
                config = pickle.dump(config,handle)
        embeddings.load("embedding_index")

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

    

    ### RETREIVE relations from KB for each query

    query2_res = {}
    query2_info = {}
    if args.task=="ai":
        queries = query_ai_uses
    elif args.task=="scifact":
        queries = queries_scifact

    for q in queries:
        print("********")
        print(q)
        x_list = []
        y_list = []
        for x in q["x"]:
            x_list.append(get_similar_spans(x))
        for y in q["y"]:
            y_list.append(get_similar_spans(y))

        x_list = pd.concat(x_list)
        if len(y_list):
            y_list = pd.concat(y_list)
        results = get_results(x_list,y_list)
        print(len(results))
        if len(results)>20:
            top = results[:10]
            bottom = results[-10:]
            sample_results = pd.concat([top,bottom])
        else:
            median = np.percentile(sample_results.avg_sim,q=50)
            top = results[results.avg_sim>=median]
            bottom = results[results.avg_sim<median]
            sample_results = pd.concat([top,bottom])

        sample_results = sample_results.sample(n=min(20,len(results)),random_state=123)
        query2_res[q["scifact_claim"]] = sample_results
        query2_info[q["scifact_claim"]] = q
        print("********")


    # Write results to file for the evaluation task

    if args.task=="ai":
        experiment_path = 'mecheffect_AI_kb_experiment_task.xlsx'
    elif args.task=="scifact"
        experiment_path = 'mecheffect_kb_experiment_task.xlsx'

    with pd.ExcelWriter(experiment_path) as writer:
        i=1
        workbook  = writer.book
        wrap_format = workbook.add_format({'text_wrap': True})
        bold = workbook.add_format({'bold': True})
        for k,v in query2_res.items():
            new_v = v.copy()
            new_v["Relevant(1=Yes,0=No)"] = None
            new_v[["x","y","context","Relevant(1=Yes,0=No)"]].to_excel(writer, startcol = 1, startrow = 2, index=False, sheet_name='search{}'.format(i))
            worksheet = writer.sheets['search{}'.format(i)]
            worksheet.set_column('A:A', 25, wrap_format)
            worksheet.set_column('B:B', 15, wrap_format)
            worksheet.set_column('C:C', 15, wrap_format)
            worksheet.set_column('D:D', 60, wrap_format)
            worksheet.set_column('E:E', 35, wrap_format)
            segments = [bold, "Research hypothesis/area: ",k]
            worksheet.write_rich_string('A1',*segments) 
            if args.task=="ai":
                query_string = "Find mechanism/effect relations where X is related to {}".format(str(query2_info[k]["x"]))
            elif args.task=="scifact"
                query_string = "Find mechanism/effect relations where X is related to {} and Y is related to {}".format(str(query2_info[k]["x"]), str(query2_info[k]["y"]))
            segments = [bold, 'Query used for results:', query_string]
            worksheet.write_rich_string('A2',*segments)
            i+=1

    # Write the umasked results (with similarity relevance scores not hidden)
    with pd.ExcelWriter("UNMASKED_"+experiment_path) as writer2:
        i=1
        for k,v in query2_res.items():
            v.to_excel(writer2, startcol = 1, startrow = 2, index=False, sheet_name='search{}'.format(i))
            worksheet = writer2.sheets['search{}'.format(i)]
            i+=1









