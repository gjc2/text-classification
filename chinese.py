import torch
import torch.nn
import jieba
import gensim
import json
from tqdm import tqdm

# 1 data processor

# 1.1 build vocab

word_list = [['<pad>', '<unk>']]
with open('cnews/cnews.train.txt', 'r', encoding='utf-8', errors='ignore') as f:
    for line in tqdm(f):
        label, content = line.strip().split('\t')
        sentence_list = [word for word in jieba.cut(content)]
        word_list.append(sentence_list)


dict = gensim.corpora.Dictionary(word_list)
print(dict.token2id)


with open('cnews/vocab.txt', 'w', encoding='utf-8') as f:
        f.write(json.dumps(dict.token2id, ensure_ascii=False))
# 1.2 data


