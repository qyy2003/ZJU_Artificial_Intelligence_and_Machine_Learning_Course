import os
import re
import pickle
import collections
import numpy as np
import jieba as jb
from mindspore.mindrecord import FileWriter

def tokenizer(docs):
    for doc in docs:
        yield re.compile(r"[A-z]{2,}(?![a-z])[A-Z][a-z]+(?=[A-Z])|[\'\w\-]+", re.UNICODE).findall(doc)
class VocabularyProcessor():
    def __init__(self, max_document_length, min_frequency):
        self.max_document_length = max_document_length
        self.min_frequency = min_frequency
        self.vocab_mapping = {'<UNK>': 0}
        self.vocab_freq = collections.defaultdict(int)
        self.tokenizer = tokenizer
        self.freeze = False
    def size(self):
        return len(self.vocab_mapping)
    def get(self, token):
        if token not in self.vocab_mapping:
            return 0
        return self.vocab_mapping[token]

    def fit(self, documents, file):
        if not self.freeze:
            for doc in self.tokenizer(documents):
                for token in doc:
                    if token != '<UNK>':
                        self.vocab_freq[token] += 1
        if self.min_frequency >0:
            for w, count in self.vocab_freq.items():
                if count > self.min_frequency:
                    self.vocab_mapping[w] = len(self.vocab_mapping)
        self.freeze = True
        with open(file, 'wb') as f:#保存字典
            pickle.dump(self.vocab_mapping, f)

    def transform(self, documents, file):
        word_id_list =[]
        if not self.freeze:
            with open(file, 'rb') as f:#加载字典
                self.vocab_mapping = pickle.load(f)
        for doc in self.tokenizer(documents):
            word_id = [0]*self.max_document_length
            for idx, token in enumerate(doc):
                if idx >= self.max_document_length:
                    break
                word_id[idx] = self.get(token)
            word_id_list.append(word_id)
        return word_id_list

def convert_to_mindrecord(features, labels, mindrecord_path):
    schema_json = {"id": {"type": "int32"},
                   "label": {"type": "int32"},
                   "feature": {"type": "int32", "shape": [-1]}}
    if not os.path.exists(mindrecord_path):
        os.makedirs(mindrecord_path)
    else:
        print(mindrecord_path, 'exists. Please make sure it is empty!')
    file_name = os.path.join(mindrecord_path, 'style.mindrecord')
    print('writing mindrecord into', file_name)
    def get_imdb_data(features, labels):
        data_list = []
        for i, (label, feature) in enumerate(zip(labels, features)):
            data_json = {"id": i,
                         "label": int(label),
                         "feature": feature.reshape(-1)}
            data_list.append(data_json)
        return data_list
    writer = FileWriter(file_name, shard_num=4)
    data = get_imdb_data(features, labels)
    writer.add_schema(schema_json, "style_schema")
    writer.add_index(["id", "label"])
    writer.write_raw_data(data)
    writer.commit()
    print('done')
