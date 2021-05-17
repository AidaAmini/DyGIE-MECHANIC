import spacy
import scispacy
import pandas as pd

from scispacy.linking import EntityLinker

def main():
  cats = set()

  input_file = open("categories_final.tsv", "r")
  for line in input_file:
    args = line.split("\t")
    cats.add(args[1])
  print(cats)
    

if __name__ == "__main__":
  main()
