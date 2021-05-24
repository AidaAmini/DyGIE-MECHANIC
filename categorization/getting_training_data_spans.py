import json

INPPUT_FILE = open("/data/aida/for_dave/data/cofie/train.json")
output_file = open("training_spans.txt", "w")
train_data = [json.loads(line) for line in INPPUT_FILE]
for item in train_data:
  for ner_item in item['ner'][0]:
    # import pdb; pdb.set_trace()

    output_file.write(" ".join(item['sentences'][0][ner_item[0]:ner_item[1]+1]) + "\n")
