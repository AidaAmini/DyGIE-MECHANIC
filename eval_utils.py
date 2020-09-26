import allennlp_models.syntax.srl
from allennlp.predictors.predictor import Predictor
import pandas as pd
import re
import itertools
import scispacy
import spacy
import numpy as np
import copy
import json
from collections import defaultdict
spacy_nlp = spacy.load('en_core_web_sm')
spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS
nlp = spacy.load("en_core_sci_sm")

from rouge import Rouge 
rouge = Rouge()

def check_contains_refrence(word):
  refrence_words = ["they", "it", "these", "those", "that", "this"]
  word_parts = word.split()
  for part in word_parts:
    if part in refrence_words:
      return True
  return False


def get_relation_scores(preds, gold, metric_list):
    res_dict = {}
    c = list(itertools.product(gold.values, preds))
    for match_metric in metric_list:
        for pair in c:
            m0 = span_score(pair[0][0], pair[1][0],match_metric) 
            m1 = span_score(pair[0][1], pair[1][1],match_metric) 
            res_dict[(pair[0][0], pair[0][1],pair[1][0], pair[1][1], match_metric)] = (m0,m1)
    return res_dict    

def get_openie_predictor():
    openiepredictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/openie-model.2020.03.26.tar.gz")
    return(openiepredictor)

def get_srl_predictor():
    import allennlp_models.syntax.srl
    srlpredictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/bert-base-srl-2020.03.24.tar.gz")
    return(srlpredictor)

def allenlp_base_relations(predictor,eval_df,four_col=False,filter_biosrl=False,collapse=True):
    uniquetext = eval_df.drop_duplicates(subset=["text"])
    print("getting predictions...")
    d = [{"sentence":t} for t in uniquetext.text.values]
    preds = predictor.predict_batch_json(d)
    
    #TODO move this to somewhere nice (json)...
    srlmap = {"treat":"MECHANISM",
    "effect":"EFFECT",
    "affect":"EFFECT",
    "caus":"EFFECT",
    "interact":"EFFECT",
    "us": "MECHANISM",
    "administ": "MECHANISM",
    "diagnos": "MECHANISM",
    "stimulat": "EFFECT",
    "inhibit": "EFFECT",
    "prevent": "MECHANISM",
    "augment": "MECHANISM",
    "accompan": "EFFECT",
    "act":"MECHANISM",
    "activate":"MECHANISM",
    "alter":"MECHANISM",
    "associat":"EFFECT",
    "bind":"MECHANISM",
    "abolish": "EFFECT",
    "abrogate": "MECHANISM",
    "block": "MECHANISM",
    "carry": "MECHANISM",
    "catalys": "MECHANISM",
    "clon": "MECHANISM",
    "begin": "MECHANISM",
    "confer": "EFFECT",
    "contain": "MECHANISM",
    "conserve": "MECHANISM",
    "control": "MECHANISM",
    "cultu": "MECHANISM",
    "decreas": "MECHANISM",
    "delet": "MECHANISM",
    "depend": "MECHANISM",
    "deriv": "MECHANISM",
    "develop": "MECHANISM",
    "differentiat": "MECHANISM",
    "disrupt": "MECHANISM",
    "regulat" : "MECHANISM",
    "eliminat" : "MECHANISM",
    "encod" : "MECHANISM",
    "enhanc" : "MECHANISM",
    "exert" : "MECHANISM",
    "express" : "EFFECT",
    "function" : "MECHANISM", 
    "generat"  : "MECHANISM",
    "includ"  : "MECHANISM",    
    "increas"  : "MECHANISM",    
    "induc" : "EFFECT",     
    "influenc" : "EFFECT",     
    "inhibit"   : "EFFECT",   
    "initiat" : "MECHANISM",
    "interact": "EFFECT",      
    "interfer" : "EFFECT",
    "involv" : "EFFECT", 
    "isolat" : "MECHANISM",     
    "lack"  : "EFFECT",    
    "lead"  :"EFFECT",   
    "link"  :"EFFECT",  
    "lose"  :"EFFECT",    
    "mediat" :"EFFECT",     
    "modify"     : "MECHANISM",  
    "modulat"   :"EFFECT",    
    "mutat"   : "MECHANISM",   
    "participat" : "MECHANISM",     
    "phosphrylat" : "MECHANISM",     
    "play"      :"EFFECT",  
    "prevent"    : "MECHANISM",  
    "produc"      : "MECHANISM",
    "proliferat"      :"EFFECT",  
    "promot"      : "MECHANISM",
    "purif"      : "MECHANISM",
    "recogniz"    : "MECHANISM",  
    "reduc"      : "MECHANISM",
    "regulat"      : "MECHANISM",
    "repress"      : "MECHANISM",
    "requir"      : "MECHANISM",
    "result"    :"EFFECT",   
    "reveal"      :"EFFECT",  
    "signal"      :"EFFECT",  
    "skip"      : "MECHANISM",
    "splic"      : "MECHANISM",
    "stimulat"     :"EFFECT",  
    "suppress"    : "MECHANISM", 
    "target"      : "MECHANISM",
    "transactivat"   : "MECHANISM",   
    "transcrib"      : "MECHANISM",
    "transfect"      : "MECHANISM",
    "transform"     : "MECHANISM", 
    "trigger"      :"EFFECT",  
    "truncat"      : "MECHANISM",      
    }

    relations =[]
    i = 0
    for sent_pred in preds:
        for v in sent_pred["verbs"]:
            rels = re.findall('\[([^]]+)', v["description"])
            relsv = [r.lstrip("V:") for r in rels if r.startswith("V")]
            rels0 = [r.lstrip("ARG0:") for r in rels if r.startswith("ARG0")]
            rels1 = [r.lstrip("ARG1:") for r in rels if r.startswith("ARG1")]
            if len(relsv) and len(rels0) and len(rels1):
                if filter_biosrl or (not collapse):
                    triggermatch = [(relsv[0],v) for k,v in srlmap.items() if (k in relsv[0])]
                    if not len(triggermatch):
                        continue
                    
                if four_col:
                    relations.append([uniquetext.iloc[i]["id"],uniquetext.iloc[i]["text"],rels0[0],rels1[0]])
                else:
                    if collapse:
                        relations.append([uniquetext.iloc[i]["id"],rels0[0],rels1[0]])
                    else:
                        # print(preds)
                        print(triggermatch[0][1])
                        relations.append([uniquetext.iloc[i]["id"],rels0[0],rels1[0],triggermatch[0][1],"1.0"])
        i+=1
    return relations

def jaccard_similarity(list1, list2):

    s1 = set(list1)
    s2 = set(list2)
    return len(s1.intersection(s2)) / len(s1.union(s2))

def exact_match(span1, span2):
    return span1.strip().lower() == span2.strip().lower()

def filter_stopwords(tokens):
    return " ".join([t for t in tokens if t.lower() not in spacy_stopwords])

def read_coref_file(coref_file_path):
  res = []
  input_file = open(coref_file_path)
  for line in input_file:
    new_row = []
    contains_refrence = False
    line_parts = line[:-1].split("\t")
    
    new_row.append(line_parts[0] + "_abstract")
    for i in range(3, len(line_parts)):

      item_parts = line_parts[i].split("|")
      if check_contains_refrence(item_parts[0]):
        contains_refrence = True
      if len(new_row) < 10:
          new_row.append(item_parts[0])

    while len(new_row) < 10:
        new_row.append("==")
    res.append(new_row)
  res_df = pd.DataFrame(res,columns =["id","arg0","arg1","arg2","arg3","arg4","arg5","arg6","arg7","arg8"])
  return res_df

def span_matching(span1,span2,metric,thresh=None):
    match = False
    if metric =="substring":
        if span1 in span2 or span2 in span1:
            return True
    elif metric =="jaccard":
        j = jaccard_similarity(span1.split(),span2.split())
        if j>thresh:
            return True
    elif metric =="head":
        doc = nlp(span1)
        root1 = [t.text for t in doc if t.dep_ =="ROOT"]
        doc = nlp(span2)
        root2 = [t.text for t in doc if t.dep_ =="ROOT"]
        if root1[0] == root2[0]:
            return True
    elif metric =="rouge":
        scores = rouge.get_scores(span1, span2)
        if scores[0]['rouge-l']['f'] > thresh:
            return True
    elif metric == "exact":
        return exact_match(span1, span2)
    return match

def span_score(span1,span2,metric):
    match = False
    if metric =="substring":
        if span1 in span2 or span2 in span1:
            return 1
        else:
            return 0
    elif metric =="jaccard":
        j = jaccard_similarity(span1.split(),span2.split())
        return j
    elif metric =="head":
        doc = nlp(span1)
        root1 = [t.text for t in doc if t.dep_ =="ROOT"]
        doc = nlp(span2)
        root2 = [t.text for t in doc if t.dep_ =="ROOT"]
        if root1[0] == root2[0]:
            return 1
        else:
            return 0
    elif metric =="rouge":
        scores = rouge.get_scores(span1, span2)
        return scores[0]['rouge-l']['f']
    elif metric == "exact":
        if exact_match(span1, span2):
            return 1
        else:
            return 0


def read_coref_matches(word_list, coref_rels):

    for row in coref_rels.iterrows():
        match_index = -1
        for i in range(9):
            if row[1]['arg' + str(i)] == word_list[0] or row[1]['arg' + str(i)] in word_list[0]:
                match_index = i
        if match_index > 0:
          for i in range(9):
            if i == match_index:
                continue
            if row[1]['arg' + str(i)] == "==":
                break
            word_list.append(word_list[0].replace(row[1]['arg' + str(match_index)],row[1]['arg' + str(i)]))
            print(row[1]['arg' + str(i)])
          
    return word_list

def relation_matching(pair, metric, coref_rels=None, labels=[1,1], thresh=0.5, filter_stop=False, span_mode=False, consider_reverse=False, coref_match=False, reverse_on_effect=False):
      #changing this so that it can check all the coref matches of the args if the oref flag is set.
      arg0match = False
      arg1match = False
      p1 = pair[0]
      p2 = pair[1]
      if type(labels[0])==str and "USED" in labels[0]:
        labels[0] = "USED"
      if type(labels[1])==str and "USED" in labels[1]:
        labels[1] = "USED"
      pair1_list_arg0 = [p1[0]]
      pair1_list_arg1 = [p1[1]]
      pair2_list_arg0 = [p2[0]]
      pair2_list_arg1 = [p2[1]]
      if type(coref_rels) ==  pd.core.frame.DataFrame:
        pair1_list_arg0 = read_coref_matches(pair1_list_arg0,coref_rels)
        pair1_list_arg1 = read_coref_matches(pair1_list_arg1,coref_rels)
        pair2_list_arg0 = read_coref_matches(pair2_list_arg0,coref_rels)
        pair2_list_arg1 = read_coref_matches(pair2_list_arg1,coref_rels)

      for pair1_arg0 in pair1_list_arg0:
        for pair1_arg1 in pair1_list_arg1:
            for pair2_arg0 in pair2_list_arg0:
                for pair2_arg1 in pair2_list_arg1:
            
                  if metric=="head":
                      filter_stop = False
                  if filter_stop:
                    p1 = [filter_stopwords(p1[0].split()),filter_stopwords(p1[1].split())]
                    p2 = [filter_stopwords(p2[0].split()),filter_stopwords(p2[1].split())]

                  if span_matching(pair1_arg0,pair2_arg0,metric,thresh):
                      arg0match = True
                      if span_matching(pair1_arg1,pair2_arg1,metric,thresh):
                          arg1match = True
                          if labels[0]==labels[1]:
                            return True
                  # considering the reverse direction for evaluating relation
                  if consider_reverse == True:
                    if reverse_on_effect == False  or ( labels[0] == "effect" or labels[0] == "EFFECT"):   #ADDED LAST : for consider reverse we want to only consider label == Effect
                        if span_matching(pair1_arg0,pair2_arg1,metric,thresh):
                          arg0match = True
                          if span_matching(pair1_arg1,pair2_arg0,metric,thresh):
                              arg1match = True
                              if labels[0]==labels[1]:
                                return True

      if span_mode:
          return (arg0match or arg1match) and labels[0]==labels[1]
      return False

def event_matching(pair, metric, coref_rels=None, labels=[1,1], thresh=0.5, filter_stop=False, span_mode=False, consider_reverse=False, coref_match=False, reverse_on_effect=False):
      arg0match = False
      arg1match = False
      p1 = pair[0]
      p2 = pair[1]

      pair1_list_arg0 = [p1[0]]
      pair1_list_arg1 = [p1[2]]
      pair2_list_arg0 = [p2[0]]
      pair2_list_arg1 = [p2[2]]
      pair1_triggers = [str(labels[0])]
      pair2_triggers = [str(labels[1])]
      for pair1_arg0 in pair1_list_arg0:          
        for pair1_arg1 in pair1_list_arg1:
            for pair2_arg0 in pair2_list_arg0:
                for pair2_arg1 in pair2_list_arg1:
            
                  if metric=="head":
                      filter_stop = False
                  if filter_stop:
                    p1 = [filter_stopwords(p1[0].split()),filter_stopwords(p1[1].split())]
                    p2 = [filter_stopwords(p2[0].split()),filter_stopwords(p2[1].split())]

                  if span_matching(pair1_arg0,pair2_arg0,metric,thresh):
                      arg0match = True
                      if span_matching(pair1_arg1,pair2_arg1,metric,thresh):
                          arg1match = True
                          if labels[0]==labels[1] or span_matching(pair1_triggers[0],pair2_triggers[0],metric,thresh):
                            return True
                  # considering the reverse direction for evaluating relation
                  if consider_reverse == True:
                    if reverse_on_effect == False  or ( labels[0] == "effect" or labels[0] == "EFFECT"):   #ADDED LAST : for consider reverse we want to only consider label == Effect
                      if span_matching(pair1_arg0,pair2_arg1,metric,thresh):
                        arg0match = True
                        if span_matching(pair1_arg1,pair2_arg0,metric,thresh):
                            arg1match = True
                            if labels[0]==labels[1] or span_matching(pair1_triggers[0],pair2_triggers[0],metric,thresh):
                              return True
      return False


def allpairs_base(golddf,pair_type="NNP",four_col=False):
    print("loading scispacy model for dep parse and NER...")
    #https://github.com/allenai/scispacy#available-models
    nlp = spacy.load("en_core_sci_sm")

    abstract2np = defaultdict(list)
    for row in golddf.drop_duplicates(subset=["id","text"]).iterrows():
        doc = nlp(row[1].text)
        for sent in doc.sents:
            if pair_type=="NNP":
                spans = [nnp.text for nnp in sent.noun_chunks]
            elif pair_type=="NER":
                spans = [ent.text for ent in sent.ents]
            elif pair_type=="joint":
                spans = [nnp.text for nnp in sent.noun_chunks] + [ent.text for ent in sent.ents]
            nnp_pairs = list(itertools.combinations(spans,2)) + list(itertools.combinations(spans[::-1],2))
            abstract2np[(row[1].id,row[1].text)]+=nnp_pairs

    relations = []
    for k,v in abstract2np.items():
        if four_col:
            _=[relations.append((k[0],k[1],m[0],m[1])) for m in v]
        else:
            _=[relations.append((k[0],m[0],m[1])) for m in v]
    return relations

def depparse_base(golddf,pair_type="NNP", four_col=False):
    relations = []
    print("loading scispacy model for dep parse and NER...")
    #https://github.com/allenai/scispacy#available-models


    uniquetext = golddf.drop_duplicates(subset=["id","text"])
    for row in uniquetext.iterrows():
        doc = nlp(row[1].text)
        
        if pair_type=="NNP":
            nps = list(doc.noun_chunks)
        elif pair_type=="NER":
            nps = list(doc.ent)

        nps = [n for n in nps if not n.root.is_stop]
        for e in nps:
            subject = None
            if e.root.dep_ in ("dobj","pobj"):
                subject = [w for w in e.root.head.lefts if w.dep_ in ["nsubj"]]
                subject = [s.text for s in subject if not s.is_stop]
                if len(subject):
                    subject = " ".join([s for s in subject])
                    matches = [subject in n.text for n in nps]
                    if len(matches):
                        matches = np.array(nps)[np.where(matches)[0]]
                        matches = [item for sublist in matches for item in sublist]
                        if len(matches):
                            if four_col:
                                _=[relations.append((row[1]["id"],row[1]["text"],m.text, e.text)) for m in matches]
                            else:
                                _=[relations.append((row[1]["id"],m.text, e.text)) for m in matches]
            if e.root.dep_ in ("nsubj"):
                subject = [w for w in e.root.head.rights if w.dep_ in ["dobj","pobj"]]
                if subject:
                    #
                    subject = [s.text for s in subject if not s.is_stop]
                    if len(subject):
                        subject = " ".join([s for s in subject])
                        matches = [subject in n.text for n in nps]
                        if len(matches):
                            matches = np.array(nps)[np.where(matches)[0]]
                            matches = [item for sublist in matches for item in sublist]
                            if len(matches):
                                if four_col:
                                    _=[relations.append((row[1]["id"],row[1]["text"],m.text, e.text)) for m in matches]
                                else:
                                    _=[relations.append((row[1]["id"],m.text, e.text)) for m in matches]

    return relations


def find_transivity_relations(rels):
    new_added = True
    seen_new = []
    while new_added:
        new_list = [x for x in rels.iterrows()]
        new_added = False
        for row1 in new_list:
            for row2 in new_list:
                if (row1[0] != row2[0]):  #we want to find transivity within same document
                    continue
                if row1[1].equals(row2[1]):
                    continue
                if row1[1]['arg1'] == row2[1]['arg0'] and (row1[1]['arg0'], row2[1]['arg1']) not in seen_new:
                  new_data = {'id': [row1[0] + ''] , 
                              'arg0': [row1[1]['arg0']],
                              'arg1': [row2[1]['arg1']]
                              }

                  if "rel" in rels.columns:
                    new_data['rel']: [row1[1]['rel']]
                  if "conf" in rels.columns:
                    new_data['conf']: [row1[1]['conf'] * row2[1]['conf']]
                  if "text" in rels.columns:
                    new_data['text']: [row1[1]['text']]
                  
                  seen_new.append((row1[1]['arg0'], row2[1]['arg1']))
                  df = pd.DataFrame(new_data).set_index("id",inplace=False)
                  rels = rels.append(df)
                  new_added = True

    return rels

def diff(relations, golddf,collapse=True, output_diff_path=None): #evaluated with exact match 
    # Function to check the changes happened to the document level for relations
    #Usage checking how many of annotations are changed /accetped without change
    good_preds = []
    not_found =[]
    seen_pred_gold = {}
    seen_pred = {}
    seen_gold = {}
    
    goldrels = golddf[["id","arg0","arg1","rel", "text"]]#.drop_duplicates()
    goldrels = goldrels.drop_duplicates(subset =["id","arg0","arg1","text"]).set_index(["id", "text"])
    predrels = relations[["id","arg0","arg1","rel","text"]].set_index(["id", "text"],inplace=False)


    for i in predrels.index.unique():
        if i in goldrels.index.unique():
            gold = goldrels.loc[i]
            if type(predrels.loc[i]) == pd.core.series.Series:
                preds = [predrels.loc[i].values]
            else:
                preds = predrels.loc[i].values
            c = list(itertools.product(gold.values, preds))
            for pair in c:
                if collapse:
                    labels = [1,1]
                else:
                    labels = [pair[0][2],pair[1][2]]

                m = relation_matching(pair,metric="exact", labels = labels, coref_rels=None)
                if m and ((i,pair[0][0],pair[0][1],pair[1][0],pair[1][1]) not in seen_pred_gold)\
                        and ((i,pair[0][0],pair[0][1]) not in seen_gold)\
                            and ((i,pair[1][0],pair[1][1]) not in seen_pred):    
                    good_preds.append([i,pair[0][0],pair[0][1]])
                    seen_pred_gold[(i,pair[0][0],pair[0][1],pair[1][0],pair[1][1])]=1
                    seen_pred[(i,pair[1][0],pair[1][1])]=1
                    seen_gold[(i,pair[0][0],pair[0][1])]=1

            seen_rels = []
            for pair in c:
                if (pair[0][0],pair[0][1]) not in seen_rels and (pair[0][0],pair[0][1]) not in seen_gold:
                    not_found.append([i[0], i[1], "", "", "", pair[0][0],pair[0][1], pair[0][2]])
                    seen_rels.append((pair[0][0],pair[0][1]))
            for pair in c:
                if (pair[1][0],pair[1][1]) not in seen_pred and (pair[1][0],pair[1][1]) not in seen_rels:
                    not_found.append([i[0], i[1], pair[1][0],pair[1][1], pair[1][2], "", "", ""])
                    seen_rels.append((pair[1][0],pair[1][1]))
            

    wrong_preds = pd.DataFrame(not_found,columns=["docid","text","arg0_pred","arg1_pred", "pred_label" ,"arg0_gold","arg1_gold", "gold_label"])
    

    common_count = 0.0
    df1 = pd.DataFrame(goldrels,columns=['text'])
    df2 = pd.DataFrame(predrels,columns=['text'])

    gold_text = [x[0][1] for x in df1.iterrows()]
    pred_text = [x[0][1] for x in df2.iterrows()]
        
    for text in gold_text:
        if text in pred_text:
          common_count += 1.0

    corr_pred = pd.DataFrame(good_preds,columns=["docid","arg0_gold","arg1_gold"])
    corr_pred = corr_pred.drop_duplicates()
    accuracy  = float(corr_pred.shape[0]/common_count)
    print("for case collase {0} agreement is : {1}".format(collapse, round(accuracy,2)))
    if output_diff_path != None:
        wrong_preds.to_csv(output_diff_path,header=True,index=False, sep="\t")


###############################################################################################
def annotation_eval(relations, golddf, coref=None, collapse = False, match_metric="substring", jaccard_thresh=0.5, transivity=False):
    goldrels = golddf[["id","arg0","arg1","rel", "text"]]#.drop_duplicates()
    goldrels = goldrels.drop_duplicates(subset =["id","arg0","arg1","text"]).set_index("id")
    predrels = relations[["id","arg0","arg1","rel","text"]].set_index("id",inplace=False)
    good_preds = []
    seen_pred_gold = {}
    seen_gold = {}
    seen_pred = {}
    for i in predrels.index.unique():
        if i in goldrels.index.unique():
            gold = goldrels.loc[i]
            if type(predrels.loc[i]) == pd.core.series.Series:
                preds = [predrels.loc[i].values]
            else:
                preds = predrels.loc[i].values
            c = list(itertools.product(gold.values, preds))
            
            for pair in c:
                if collapse:
                    labels = [1,1]
                else:
                    labels = [pair[0][2],pair[1][2]]

                m = relation_matching(pair,metric=match_metric, labels = labels,thresh=jaccard_thresh,coref_rels=None)
                if m and ((i,pair[0][0],pair[0][1],pair[1][0],pair[1][1]) not in seen_pred_gold)\
                        and ((i,pair[0][0],pair[0][1]) not in seen_gold)\
                            and ((i,pair[1][0],pair[1][1]) not in seen_pred):    
                    good_preds.append([i,pair[0][0],pair[0][1]])
                    seen_pred_gold[(i,pair[0][0],pair[0][1],pair[1][0],pair[1][1])]=1
                    seen_pred[(i,pair[1][0],pair[1][1])]=1
                    seen_gold[(i,pair[0][0],pair[0][1])]=1

    common_count = 0.0
    df1 = pd.DataFrame(goldrels,columns=['text'])
    df2 = pd.DataFrame(predrels,columns=['text'])

    gold_text = [x[1]['text'] for x in df1.iterrows()]
    pred_text = [x[1]['text'] for x in df2.iterrows()]
        
    for text in gold_text:
        if text in pred_text:
          common_count += 1.0
    for text in pred_text:
        if text in gold_text:
          common_count += 1.0

    if common_count == 0:
        return 0
    corr_pred = pd.DataFrame(good_preds,columns=["docid","arg0_gold","arg1_gold"])
    corr_pred = corr_pred.drop_duplicates()
    return 2*float(corr_pred.shape[0]/common_count)

###########################################################################################################
def ie_span_eval(relations, golddf, match_metric="substring", jaccard_thresh=0.5):
    goldrels = golddf[["id","arg0","arg1","rel"]]#.drop_duplicates()
    goldrels = goldrels.drop_duplicates(subset =["id","arg0","arg1"]).set_index("id")
    if "conf" in relations.columns:
        predrels = relations[["id","arg0","arg1","rel","conf"]].set_index("id",inplace=False)
    else:
        predrels = relations[["id","arg0","arg1"]].set_index("id",inplace=False)
    good_preds = []
    gold_spans = []
    pred_spans = []
    found_from_gold = []
    seen_pred_gold = {}
    for i in predrels.index.unique():
        if i in goldrels.index.unique():
            gold = goldrels.loc[i]
            if type(predrels.loc[i]) == pd.core.series.Series:
                preds = [predrels.loc[i].values]
            else:
                preds = predrels.loc[i].values
            c = list(itertools.product(gold.values, preds))
            for pair in c:
            
                for pair0_ind in range(0,2):
                    for pair1_ind in range(0,2):
                        #adding gold and pred spans to have count for final calculations
                        if [i,pair[1][pair1_ind]] not in pred_spans:
                            pred_spans.append([i,pair[1][pair1_ind]])
                        if [i,pair[0][pair0_ind]] not in gold_spans:
                            gold_spans.append([i,pair[0][pair0_ind]])

                        m = span_matching(pair[0][pair0_ind], pair[1][pair1_ind],match_metric,thresh=jaccard_thresh)   
                        if m and ((i,pair[0][pair0_ind],pair[1][pair1_ind]) not in seen_pred_gold):
                            if [i,pair[1][pair1_ind]] not in good_preds:
                                good_preds.append([i,pair[1][pair1_ind]])
                            if [i,pair[0][pair0_ind]] not in found_from_gold:
                                found_from_gold.append([i,pair[0][pair0_ind]])
                            seen_pred_gold[(i,pair[0][pair0_ind],pair[1][pair1_ind])]=1
    
    corr_pred = pd.DataFrame(good_preds,columns=["docid","arg0_gold"])
    corr_gold = pd.DataFrame(found_from_gold,columns=["docid","arg0_gold"])
    corr_pred = corr_pred.drop_duplicates()
    corr_gold = corr_gold.drop_duplicates()

    goldspans = pd.DataFrame(gold_spans,columns=["docid","arg0_gold"])
    goldspans = goldspans.drop_duplicates()
    predspans = pd.DataFrame(pred_spans,columns=["docid","arg0_gold"])
    predspans = predspans.drop_duplicates()

    TP = corr_pred.shape[0]
    TP_for_recal = corr_gold.shape[0]
    FP = predspans.shape[0] - TP
    FN = goldspans.shape[0] - TP_for_recal

    precision = TP/(TP+FP)
    recall = TP_for_recal/(FN+TP_for_recal)

    F1 = 2*(precision * recall) / (precision + recall)

    return corr_pred, precision,recall, F1

def ie_eval_agreement(relations, golddf, coref=None, collapse = False, match_metric="substring", jaccard_thresh=0.5, transivity=True, topK=None, consider_reverse=False):
    # A fuunction called for calculation of f1 agreememt between common annotations.
    #cdifference here is that the gold has some relations that are not in pred. so we should remove those 
    goldrels = golddf[["id","arg0","arg1","rel","text"]]#.drop_duplicates()
    goldrels = goldrels.drop_duplicates(subset =["id","arg0","arg1","text"]).set_index(["id", "text"])
    predrels = relations[["id","arg0","arg1","rel","text"]].set_index(["id", "text"],inplace=False)

    gold_count = 0
    pred_count = 0

    good_preds = []
    found_from_gold = []
    seen_pred_gold = {}
    if topK == None or topK > len(predrels)-1:
        topK = len(predrels)
    
    predrels = predrels[:topK]
    for i in predrels.index.unique():
        if i in goldrels.index.unique():
            gold = goldrels.loc[[i]]
            if type(predrels.loc[i]) == pd.core.series.Series:
                preds = [predrels.loc[i].values]
            else:
                preds = predrels.loc[i].values

            gold_count += gold.shape[0]
            try:
                pred_count += preds.shape[0]
            except:
                pred_count += preds[0].shape[0]

            c = list(itertools.product(gold.values, preds))
            for pair in c:
                if collapse:
                    labels = [1,1]
                else:
                    labels = [pair[0][2],pair[1][2]]

                m = relation_matching(pair,metric=match_metric, labels = labels,thresh=jaccard_thresh,coref_rels=None,consider_reverse=consider_reverse)
                if m and pair[0][2] != pair[1][2].replace("USED-TO", "USED"):
                  print( i[1] + '\t' + pair[0][0] + "\t" + pair[0][1] + "\t" + pair[0][2] + "\t" + pair[1][0] + '\t' + pair[1][1] + "\t" + pair[1][2])

                if m and ((i,pair[0][0],pair[0][1],pair[1][0],pair[1][1]) not in seen_pred_gold):
                    if [i,pair[1][0],pair[1][1]] not in good_preds:
                        good_preds.append([i,pair[1][0],pair[1][1]])
                    if [i,pair[0][0],pair[0][1]] not in found_from_gold:
                        found_from_gold.append([i,pair[0][0],pair[0][1]])
                    seen_pred_gold[(i,pair[0][0],pair[0][1],pair[1][0],pair[1][1])]=1
    
    corr_pred = pd.DataFrame(good_preds,columns=["docid","arg0_gold","arg1_gold"])
    corr_gold = pd.DataFrame(found_from_gold,columns=["docid","arg0_gold","arg1_gold"])
    corr_pred = corr_pred.drop_duplicates()
    corr_gold = corr_gold.drop_duplicates()
    
    TP = corr_pred.shape[0]
    TP_for_recal = corr_gold.shape[0]
    FP = pred_count - TP
    FN = gold_count - TP_for_recal

    precision = TP/(TP+FP)
    recall = TP_for_recal/(FN+TP_for_recal)

    if recall==0 and precision == 0:
        F1 = 0
    else:
        F1 = 2*(precision * recall) / (precision + recall)

    return corr_pred, precision,recall, F1

def ie_eval(relations, golddf, coref=None, collapse = False, match_metric="substring", jaccard_thresh=0.5, transivity=True, topK=None, consider_reverse=False,reverse_on_effect=False):
    
    goldrels = golddf[["id","arg0","arg1","rel"]]#.drop_duplicates()
    goldrels = goldrels.drop_duplicates(subset =["id","arg0","arg1"]).set_index("id")

    if type(coref) ==pd.core.frame.DataFrame:
        corefrels = coref.set_index("id")
    #only get rel for our model / gold, otherwise assume one collapsed label
    if "conf" in relations.columns:
        predrels = relations[["id","arg0","arg1","rel","conf"]].set_index("id",inplace=False)
        predrels = predrels.sort_values(by='conf',ascending=False)
    else:
        predrels = relations[["id","arg0","arg1"]].set_index("id",inplace=False)

    
    if transivity:
        goldrels_trans = find_transivity_relations(goldrels)
    else:
        goldrels_trans = goldrels

    good_preds = []
    found_from_gold = []
    seen_pred_gold = {}
    if topK == None or topK > len(predrels)-1:
        topK = len(predrels)
    
    # finding the rels that exist in the preds
    predrels = predrels[:topK]
    for i in predrels.index.unique():
        if i in goldrels_trans.index.unique():
            gold = goldrels_trans.loc[[i]]
            coref_rels = None

            if type(coref) == pd.core.frame.DataFrame and i in corefrels.index.unique():
                coref_rels = corefrels.loc[[i]]
            if type(predrels.loc[i]) == pd.core.series.Series:
                preds = [predrels.loc[i].values]
            else:
                preds = predrels.loc[i].values
            c = list(itertools.product(gold.values, preds))
            for pair in c:
                if collapse:
                    labels = [1,1]
                else:
                    labels = [pair[0][2],pair[1][2]]

                m = relation_matching(pair,metric=match_metric,labels=labels,thresh=jaccard_thresh,coref_rels=coref_rels,consider_reverse=consider_reverse,reverse_on_effect=reverse_on_effect)
                if m and ((i,pair[0][0],pair[0][1],pair[1][0],pair[1][1]) not in seen_pred_gold):
              
                    if [i,pair[1][0],pair[1][1]] not in good_preds:
                        good_preds.append([i,pair[1][0],pair[1][1]])
                    seen_pred_gold[(i,pair[0][0],pair[0][1],pair[1][0],pair[1][1])]=1

    #checking for the ones that are found from the gold.
    seen_pred_gold = {}
    for i in predrels.index.unique():
        if i in goldrels.index.unique():
            gold = goldrels.loc[[i]]
            coref_rels= None
            if type(coref) == pd.core.frame.DataFrame and i in corefrels.index.unique():
                coref_rels = corefrels.loc[[i]]
            if type(predrels.loc[i]) == pd.core.series.Series:
                preds = [predrels.loc[i].values]
            else:
                preds = predrels.loc[i].values
            c = list(itertools.product(gold.values, preds))
            for pair in c:
                if collapse:
                    labels = [1,1]
                else:
                    labels = [pair[0][2],pair[1][2]]
                m = relation_matching(pair,metric=match_metric, labels = labels,thresh=jaccard_thresh,coref_rels=coref_rels,consider_reverse=consider_reverse)
                if m and ((i,pair[0][0],pair[0][1],pair[1][0],pair[1][1]) not in seen_pred_gold):
                    if [i,pair[0][0],pair[0][1]] not in found_from_gold:
                        found_from_gold.append([i,pair[0][0],pair[0][1]])
                    seen_pred_gold[(i,pair[0][0],pair[0][1],pair[1][0],pair[1][1])]=1
    
    corr_pred = pd.DataFrame(good_preds,columns=["docid","arg0_gold","arg1_gold"])
    corr_gold = pd.DataFrame(found_from_gold,columns=["docid","arg0_gold","arg1_gold"])
    corr_pred = corr_pred.drop_duplicates()
    corr_gold = corr_gold.drop_duplicates()
    

    TP = corr_pred.shape[0]
    TP_for_recal = corr_gold.shape[0]
    FP = topK - TP
    FN = goldrels.shape[0] - TP_for_recal

    precision = TP/(TP+FP)
    recall = TP_for_recal/(FN+TP_for_recal)

    F1 = 2*(precision * recall) / (precision + recall)
    return corr_pred, precision,recall, F1

#########################################################################################
def ie_eval_event(relations, golddf, coref=None, collapse = False, match_metric="substring", jaccard_thresh=0.5, transivity=False, topK=None, consider_reverse=False):
    goldrels = golddf[["id","arg0","trigger","arg1"]]#.drop_duplicates()
    goldrels = goldrels.drop_duplicates(subset =["id","arg0","trigger","arg1"]).set_index("id")

    if type(coref) ==pd.core.frame.DataFrame:
        corefrels = coref.set_index("id")
    predrels = relations[["id","arg0","trigger","arg1"]].set_index("id",inplace=False)

    good_preds = []
    found_from_gold = []
    seen_pred_gold = {}
    if topK == None or topK > len(predrels)-1:
        topK = len(predrels)
    
    predrels = predrels[:topK]
    for i in predrels.index.unique():
        if i in goldrels.index.unique():
            gold = goldrels.loc[[i]]
            coref_rels = None

            if type(coref) == pd.core.frame.DataFrame and i in corefrels.index.unique():
                coref_rels = corefrels.loc[[i]]
            if type(predrels.loc[i]) == pd.core.series.Series:
                preds = [predrels.loc[i].values]
            else:
                preds = predrels.loc[i].values
            c = list(itertools.product(gold.values, preds))
            
            for pair in c:
                if collapse:
                    labels = [1,1]
                else:
                    labels = [pair[0][1],pair[1][1]]

                m = event_matching(pair,metric=match_metric,labels=labels,thresh=jaccard_thresh,coref_rels=coref_rels,consider_reverse=consider_reverse)
                if m and ((i,pair[0][0],pair[0][2],pair[1][0],pair[1][2]) not in seen_pred_gold):
                    if len(pair[0][0]) == 1 or len(pair[1][0]) == 1:
                        continue
                    if [i,pair[1][0],pair[1][2]] not in good_preds:
                        good_preds.append([i,pair[1][0],pair[1][1],pair[1][2]])
                    if [i,pair[0][0],pair[0][1],pair[0][2]] not in found_from_gold:
                        found_from_gold.append([i,pair[0][0],pair[0][1],pair[0][2]])
                    seen_pred_gold[(i,pair[0][0],pair[0][1],pair[0][2],pair[1][0],pair[1][1],pair[1][2])]=1
    
                    
    corr_pred = pd.DataFrame(good_preds,columns=["docid","arg0_gold","trig_gold","arg1_gold"])
    corr_gold = pd.DataFrame(found_from_gold,columns=["docid","arg0_gold","trig_gold","arg1_gold"])
    corr_pred = corr_pred.drop_duplicates()
    corr_gold = corr_gold.drop_duplicates()
    

    TP = corr_pred.shape[0]
    TP_for_recal = corr_gold.shape[0]
    FP = topK - TP
    FN = goldrels.shape[0] - TP_for_recal

    precision = TP/(TP+FP)
    recall = TP_for_recal/(FN+TP_for_recal)

    F1 = 2*(precision * recall) / (precision + recall)
    return corr_pred, precision,recall, F1


#########################################################################################


def ie_errors(relations, golddf, coref=None, collapse = False, match_metric="substring", jaccard_thresh=0.5, transivity=True, topK=None):
   
    # finding and printing the cases that are not matched for error analysis

    goldrels = golddf[["id","arg0","arg1","rel"]]#.drop_duplicates()
    goldrels = goldrels.drop_duplicates(subset =["id","arg0","arg1"]).set_index("id")

    gold_id_text = golddf[["id","text"]]
    gold_id_text = gold_id_text.drop_duplicates(subset =["id","text"]).set_index("id")



    if coref != None:
        corefrels = coref.set_index("id")
    #only get rel for our model / gold, otherwise assume one collapsed label
    if "conf" in relations.columns:
        predrels = relations[["id","arg0","arg1","rel","conf"]].set_index("id",inplace=False)
        predrels = predrels.sort_values(by='conf',ascending=False)
    else:
        predrels = relations[["id","arg0","arg1"]].set_index("id",inplace=False)

    goldrels_trans = goldrels
    if transivity:
        goldrels_trans = find_transivity_relations(goldrels)
        predrels_trans = find_transivity_relations(predrels)

    not_found = []
    pred_matched = []
    gold_mathced = []
    seen_pred_gold = {}
    for i in predrels.index.unique():
        if i in goldrels_trans.index.unique():
        
            gold = goldrels_trans.loc[[i]]
            gold_text_id = gold_id_text.loc[[i]]
            coref_rels = None
            if coref != None:
                coref_rels = corefrels.loc[[i]]
            if type(predrels.loc[i]) == pd.core.series.Series:
                preds = [predrels.loc[i].values]
            else:
                preds = predrels.loc[i].values
            c = list(itertools.product(gold.values, preds))
            found_count = 0
            for pair in c:
                if collapse:
                    labels = [1,1]
                else:
                    labels = [pair[0][2],pair[1][2]]
                m = relation_matching(pair,metric=match_metric, labels = labels,thresh=jaccard_thresh,coref_rels=coref_rels)
                #changing this so that it can check all the coref matches of args.if m and
                if m and ((i,pair[0][0],pair[0][1],pair[1][0],pair[1][1]) not in seen_pred_gold):
                    gold_mathced.append((pair[0][0],pair[0][1]))
                    pred_matched.append((pair[1][0],pair[1][1]))
                    seen_pred_gold[(i,pair[0][0],pair[0][1],pair[1][0],pair[1][1])]=1
                    found_count += 1

            # only add relations of the predictions samples that are not doing good
            if float(found_count/len(preds)) < 0.5:
                seen_rels = []
                for pair in c:
                    if (pair[0][0],pair[0][1]) not in seen_rels:
                        not_found.append([i, gold_text_id.iloc[0]["text"], "", "", "", pair[0][0],pair[0][1], pair[0][2]])
                        seen_rels.append((pair[0][0],pair[0][1]))
                
                for pair in c:
                    if (pair[1][0],pair[1][1]) not in pred_matched and (pair[1][0],pair[1][1]) not in seen_rels:
                        not_found.append([i,gold_text_id.iloc[0]["text"], pair[1][0],pair[1][1], pair[1][2], "", "", ""])
                        seen_rels.append((pair[1][0],pair[1][1]))
                
    
    wrong_preds = pd.DataFrame(not_found,columns=["docid","text","arg0_pred","arg1_pred", "pred_label" ,"arg0_gold","arg1_gold", "gold_label"])
    
    return wrong_preds


