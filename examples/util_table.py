import json

f = open('/root/data/table/1_train.jsonl')
line = f.readline()
line = json.loads(line)
table = json.loads(line['table']['raw_json'])

rows = table['data']
cols = [table['title']] + table['data']
cols = list(map(list, zip(*cols)))


import fasttext

embed_path = '/root/data/word_embedding/wiki.simple.bin'

model = fasttext.FastText.load_model(embed_path)
model.get_word_vector('w')
