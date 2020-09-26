import pandas as pd
def get_similar_spans(embeddings, uniqueterms, query,threshold=0.75):
    """
    Perform approximate nearest neighbors search to find similar spans to query
    """
    similar_span_list = []
    hits = embeddings.search(query,limit=10000)
    high = [(uniqueterms[h[0]],h[1]) for h in hits if h[1]>=threshold]
    
    similar_spans = pd.DataFrame(high,columns=["match","similarity"])
    similar_spans["sim_rank"] = [1]*len(high)

    similar_span_list.append(similar_spans)
    similar_spans = pd.concat(similar_span_list)
    similar_spans = similar_spans.drop_duplicates(subset=["match"])
    return similar_spans

def get_sim(row,xy,bothxy=False):
    """
    Helper function adding query similarity to each relation
    """
    row["x_sim"] = xy.loc[row.x].similarity
    row["x_simrank"] =  xy.loc[row.x].sim_rank
    row["avg_sim"] = row["x_sim"]
    if bothxy:
        print("bothxy")
        row["y_sim"] = xy.loc[row.y].similarity
        row["y_simrank"] =  xy.loc[row.y].sim_rank
        row["avg_sim"] = (row["x_sim"]+row["y_sim"])/2
    return row 
    
def get_retrieved_rels_table(x_list,y_list, rels,both=True):
    """
    Get table with search results
    """
    rels.drop_duplicates(subset=["sentence","norm_span1","norm_span2"],inplace=True)
    rels = rels[["norm_span1","norm_span2","sentence"]]
    rels.columns = ["x", "y","context"]
    if len(y_list):
        if len(x_list):
            bothxy = True
            xy = pd.concat([x_list, y_list]).drop_duplicates("match").set_index("match")
        else:
            bothxy = False
            xy = pd.concat([y_list]).drop_duplicates("match").set_index("match")
    
    else:
        bothxy = False
        xy = x_list.drop_duplicates("match").set_index("match")

    rels = rels.apply(get_sim,axis=1,args=(xy,bothxy,))
    rels = rels.sort_values("avg_sim",ascending=False)
    return rels   


def get_results(q, kb, x_list,y_list,threshold = 0.75):
    """
    #Get KB table
    #All rows in kb where span1 in x_list, span2 in y_list, where 
    """
    if len(x_list) and len(y_list):
        print("both conditions")
        if q["bidirect"]:
            rels1 = kb[(kb.norm_span1.isin(x_list.match.values)) & (kb.norm_span2.isin(y_list.match.values))]
            rels2 = kb[(kb.norm_span2.isin(x_list.match.values)) & (kb.norm_span1.isin(y_list.match.values))]
            rels = pd.concat([rels1,rels2])
            results = get_retrieved_rels_table(x_list,y_list,rels)
        else:
            rels = kb[(kb.norm_span1.isin(x_list.match.values)) & (kb.norm_span2.isin(y_list.match.values))]
            results = get_retrieved_rels_table(x_list,y_list,rels)
    elif len(x_list):
        print("x")
        rels = kb[(kb.norm_span1.isin(x_list.match.values))]
        results = get_retrieved_rels_table(x_list,y_list,rels)

    elif len(y_list):
        print("y")
        rels = kb[(kb.norm_span2.isin(y_list.match.values))]
        results = get_retrieved_rels_table(x_list,y_list,rels)

    return results
