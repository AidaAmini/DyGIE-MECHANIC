import spacy
import scispacy
import pandas as pd


"""
Input: tui_filename (file name with tag to category name)
Create dict of tag to category name based on file
"""
def read_type_ids(tui_filename):
  tui_label_dict = {}
  input_file = open(tui_filename)
  for line in input_file:
    line_parts = line[:-1].split("\t")
    tui_label_dict[line_parts[0]] = line_parts[1]
  return tui_label_dict

def get_cat(doc):
  top_cat = "None"
  top_prob = 0
  for cat in categories:
    doc2 = nlp(cat)
    prob = doc.similarity(doc2)
    if (top_prob < prob):
      top_cat = cat
      top_prob = prob
  assert top_cat != "None"
  return top_cat, top_prob


"""
Generate categories from previous categories
"""
def generate_cats(output_name):
  output_file = open(output_name, "w")

  for value in tui_label_dict.values():
    doc = nlp(value)
    cat, prob = get_cat(doc)
    output_file.write(value + "\t" + cat + "\t" + str(prob) + "\n")

PATH_TO_DATA = "../for_edan/"
tui_label_dict = read_type_ids(PATH_TO_DATA + "tui_labels.tsv")

categories = ["Animal", "Organism", "Drug", "Cell", "Behavior", "Organization",
                "Group", "Disease", "Symptom", "Body Part", "Organ", "Medical Device",
                "Product", "Concept", "Finding", "Procedure", "Molecule", "Organic Chemical", "Virus"]

if __name__ == "__main__":
  # activated = spacy.prefer_gpu()
  # if activated is None:
  #   print("No GPU specified")
  # else:
  #   print("Using GPU")
  nlp = spacy.load("en_core_sci_lg")
  generate_cats("categories_temp.tsv")