from transformers import BertTokenizer, BertModel
from joblib import load

class Globals():
	MODEL_VERSION = 'allenai/scibert_scivocab_uncased'
	DO_LOWER_CASE = True
	MODEL = BertModel.from_pretrained(MODEL_VERSION)
	TOKENIZER = BertTokenizer.from_pretrained(MODEL_VERSION, do_lower_case=DO_LOWER_CASE)
	KMEANS = load("/data/edan/categorization/nosymbol_revmean_10_kmeans/revised_kmeans.joblib")
	CLUSTER_SIZE = 10
	TYPE = "mean"
	NAME = "nosymbol_revmean_10"
	MAP = None