import json 
from pathlib import Path
import os
import json
import jsonlines
import glob
import sys
from pathlib import Path
from tqdm import tqdm
import argparse

"""
Usage
python scripts/data/schema_mapper.py --dataroot ../coviddata --dataset covid_anno_par --map_type mech
"""

def load_map_dict(map_path):
    with open(map_path,"r") as f:
        schemamap = json.load(f)
    return schemamap

def map_ner(doc_dat):
    """
    Maps any NER class name to "ENTITY"
    """
    new_ner = []
    for ner_list in doc_dat['ner']:
        new_ner_list = []
        for ner in ner_list:
          ner[2] = "ENTITY"
          new_ner_list.append(ner)
        new_ner.append(new_ner_list)
    doc_dat['ner']  = new_ner

def map_relation(doc_dat,schemamap):
    """
    Maps relation classes to new schema given by dict
    """
    new_rel = []

    for rel_list in doc_dat['relations']:
        new_rel_list = []
        #rel_list = [rel for rel in rel_list if rel[4] in schemamap]
        for rel in rel_list:
            if rel[4] in schemamap:
                rel[4] = schemamap[rel[4]]    
                new_rel_list.append(rel)
        new_rel.append(new_rel_list)
    
    doc_dat['relations'] =  new_rel

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, default='../coviddata', help='root dir for dataset folders')
    parser.add_argument('--dataset', type=str, choices = ['scierc','covid_anno_par','covid_anno_augmented_par','chemprot','srl'], default='chemprot', help='which dataset to map')
    parser.add_argument('--maptype', choices = ['mech','mech_effect'],default="mech", help='map to mech only or to mech effect')

    args = parser.parse_args()

    root_path = Path(args.dataroot)
    dataset_dir = root_path / "UnifiedData"/ args.dataset
    original_dir = dataset_dir / "original"
    map_dir = dataset_dir / "mapped"

    map_type = args.maptype
    map_path = map_dir.joinpath(map_type+".txt")

    schemamap = load_map_dict(map_path)   
    original_files = list(original_dir.glob('*.jsonl'))
    original_files.extend(list(original_dir.glob('*.json')))

    fold_mapped_dir = map_dir.joinpath(map_type)
    Path(fold_mapped_dir).mkdir(parents=True, exist_ok=True)
    print("--- loading and mapping from ", original_dir)
    for fold in original_files:
        print(fold.name)
        fold_mapped_dir = map_dir.joinpath(map_type)
        Path(fold_mapped_dir).mkdir(parents=True, exist_ok=True)
        fold_mapped = fold_mapped_dir/fold.name
        new_jsons = []
        with jsonlines.open(fold,'r') as reader:
            for obj in tqdm(reader):
                map_ner(obj)
                map_relation(obj,schemamap)
                relations_size = 0
                for rel_list in obj["relations"]:
                    relations_size += len(rel_list)
                if relations_size > 0:
                    new_jsons.append(obj)
                else:
                    new_jsons.append(obj)
                    print("removed")

        with jsonlines.open(str(fold_mapped).rstrip("l"), 'w') as writer:
            writer.write_all(new_jsons)
