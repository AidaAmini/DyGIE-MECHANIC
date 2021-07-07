import spacy
import scispacy
import pandas as pd

from scispacy.linking import EntityLinker
BLENDER_REL_FILE_LISTS = ["genes_diseases_relation.csv","chem_gene_ixns_relation.csv", "chemicals_diseases_relation.csv" ]

PATH_TO_DATA = "../../for_edan/"

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

"""
Input: file_path (file name)
copy lines of file to elements of list but remove newline
"""
def read_temp_file(file_path):
  input_file = open(file_path)
  res = []
  for line in input_file:
    res.append(line[:-1])
  return res

"""
Depricated
"""
def write_cofie_entities(cofie_kb_path):
    seen_spans = []
    output_file = open(PATH_TO_DATA + "cofie_spans.txt", "w")
    input_file = open(cofie_kb_path)
    count = 0
    for line in input_file:

      if count % 10000 == 0:
        print(count)
      count += 1
      
      if count == 1:
        continue
      line_parts = line[:-1].split("\t")
      # import pdb; pdb.set_trace()
      for i in [2,3]:
        if line_parts[i] not in seen_spans:
          output_file.write(line_parts[i] + '\n')
          seen_spans.append(line_parts[i])
          
"""
Depricated
"""
def write_blender_entities(blender_kb_path):
    # seen_spans = []
    seen_spans = read_temp_file("blender_spans_temp.txt")
    output_file = open("blender_spans.txt", "w")
    for span in seen_spans:
      output_file.write(span + '\n')
    for file in BLENDER_REL_FILE_LISTS:
      df = pd.read_csv(blender_kb_path + file, header=0, sep="\t")

      for row in df.iterrows():
        if file == "genes_diseases_relation.csv" and int(row[0]) < 43700000:
          continue
        if row[0] % 100000 == 0:
          print(file)
          print(str(row[0]))
        if file == "genes_diseases_relation.csv":
          span1 = str(row[1]["GeneSymbol"]).lower()
          span2 = str(row[1]["DiseaseName"]).lower()
        elif file == "chem_gene_ixns_relation.csv":
          span1 = str(row[1]["ChemicalName"]).lower()
          span2 = str(row[1]["GeneName"]).lower()
        else:
          span1 = str(row[1]["ChemicalName"]).lower()
          span2 = str(row[1]["DiseaseName"]).lower()

        if span1 not in seen_spans:
          seen_spans.append(span1)
          output_file.write(span1 + '\n')

        if span2 not in seen_spans:
          seen_spans.append(span2)
          output_file.write(span2 + '\n')

 
"""
Input: input_filename 
       output_filename
"""
def get_types(input_filename, output_filename):
    nlp = spacy.load("en_core_sci_sm")
    type_count_dict = {}
    log_file = open("blender_log.tsv", "w")
    # This line takes a while, because we have to download ~1GB of data
    # and load a large JSON file (the knowledge base). Be patient!
    # Thankfully it should be faster after the first time you use it, because
    # the downloads are cached.
    # NOTE: The resolve_abbreviations parameter is optional, and requires that
    # the AbbreviationDetector pipe has already been added to the pipeline. Adding
    # the AbbreviationDetector pipe and setting resolve_abbreviations to True means
    # that linking will only be performed on the long form of abbreviations.
    tui_label_dict = read_type_ids(PATH_TO_DATA + "tui_labels.tsv")
    linker = EntityLinker(resolve_abbreviations=True, name="umls")
    nlp.add_pipe(linker)

    input_file = open(input_filename)
    count = 0
    for line in input_file:
      count += 1
      if count % 1000 == 0:
        print("count " + str(count))
      doc = nlp(line[:-1])
      if len(doc.ents) != 0:
        for i in range(len(doc.ents)):
          entity = doc.ents[i]
          for umls_ent in entity._.kb_ents:

            types_list = linker.kb.cui_to_entity[umls_ent[0]].types
            for typ in types_list:
              if typ not in type_count_dict:
                type_count_dict[typ] = 0
              type_count_dict[typ] += 1
              # import pdb; pdb.set_trace()
              log_file.write(line[:-1] + "\t" + tui_label_dict[typ]+ "\n")
    output_file = open(output_filename, "w")
    for typ in type_count_dict:
      output_file.write(typ + "\t" + str(type_count_dict[typ]) + '\n')

      

# write_cofie_entities("/home/aida/covid_clean/dygiepp/complete_KB_coref.tsv")
# write_blender_entities("/data/aida/")
#get_types("blender_spans2.txt", "blender_spans_tags.txt")
get_types(PATH_TO_DATA + "cofie_spans.txt", "cofie_spans_tags.txt")



