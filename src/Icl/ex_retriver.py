import faiss
import json
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import random
import numpy as np
from tqdm import tqdm
import sys
sys.path.append("..")
from gnnencoder.encoder import SentenceEncoder

class Ex_Retriver():
    def __init__(self, ex_file, paths=None, encode_method='bert'):
        '''
        input: ex_file: 需要构建检索的例子文件（一般为原始训练集）
        '''
        self.encode_method = encode_method
        self.selected_k = 3

        with open(ex_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.sents = []
            self.labels = []
            for d in data:
                self.sents.append(d['input'])
                self.labels.append(d['output'])
        self.data_dict = {}
        for sent, label in zip(self.sents, self.labels):
            self.data_dict[sent] = label

        # Initialize different models based on the specified method
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if encode_method == 'bert':
            self.model = AutoModel.from_pretrained(paths['bert_path']).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(paths['bert_path'])
            self.init_embeddings(self.sents)
        elif encode_method == 'gnn':
            gnn_model_path = paths['gnn_path']
            bert_model_path = paths['bert_path']
            self.gnn_encoder = SentenceEncoder(gnn_model_path, bert_model_path)
            self.init_embeddings(self.sents)
        elif encode_method == 'random':
            pass
        else:
            raise NotImplementedError
        
    def encode_sentences(self, sents, batch_size=512):
        '''
        sents: 所有需要编码的句子
        '''
        if self.encode_method == 'bert':
            return self.bert_encode_sentences(sents)
        elif self.encode_method == 'gnn':
            return self.gnn_encode_sentences(sents)
        elif self.encode_method is None:
            return None
        else:
            raise NotImplementedError
        
    def gnn_encode_sentences(self, sents, batch_size=128):
        '''
        sents: 所有需要编码的句子
        '''
        all_dims_embeddings = [[], []]

        for i in range(0, len(sents), batch_size):
            # 每个维度分别构建检索
            batch_sents = sents[i:i + batch_size]
            dims_representations, avg_representation = self.gnn_encoder.encode(batch_sents)
            avg_embeddings = avg_representation.cpu().numpy()

            for i in range(1):
                all_dims_embeddings[i].append(dims_representations[i].cpu().numpy())
            all_dims_embeddings[1].append(avg_embeddings)

        # return np.concatenate(all_embeddings, axis=0)
        return [np.concatenate(all_dims_embeddings[i], axis=0) for i in range(2)]

    def bert_encode_sentences(self, sents, batch_size=512):
        '''
        sents: 所有需要编码的句子
        batch_size: 每次编码的batch size
        '''
        all_embeddings = []

        for i in range(0, len(sents), batch_size):
            batch_sents = sents[i:i + batch_size]
            encoded_input = self.tokenizer(batch_sents, padding=True, truncation=True, return_tensors='pt')
            encoded_input = {key: val.to(self.device) for key, val in encoded_input.items()}  # Move input to device

            with torch.no_grad():
                model_output = self.model(**encoded_input)

            embeddings = model_output[0][:, 0, :]

            embeddings = F.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.cpu().numpy())

        return np.concatenate(all_embeddings, axis=0)

    def init_embeddings(self, sents):
        print("Initializing embeddings...")
        # build the index using FAISS
        embeddings = self.encode_sentences(sents)
        if self.encode_method == 'gnn':
            # 针对每个特征维度分别构建检索
            d = embeddings[0].shape[1]
            self.index = [faiss.IndexFlatL2(d) for i in range(3)]
            for i in range(2):
                self.index[i].add(embeddings[i])
        else:
            d = embeddings.shape[1]

            self.index = faiss.IndexFlatL2(d)
            self.index.add(embeddings)

    def search_examples(self, query, selected_k=3, verbose=False):
        if self.encode_method == 'random':
            return random.sample(list(zip(self.data_dict.keys(), self.data_dict.values())), selected_k)
        
        if verbose:
            print(f"\nSearching for: {query}")

        if selected_k is None:
            selected_k = self.selected_k

        if self.encode_method == 'gnn':
            query_embeddings = self.encode_sentences([query])
            choosed_idxs = {} # 已选择的索引
            # 每个维度要选择的数量
            n_dims = [2, 1]
            feture_types = ['lig', 'avg']
            for i in range(2):
                distances, indices = self.index[i].search(query_embeddings[i], self.index[i].ntotal)
                # 距离越小的放到前面（这样最相似的例子离输入最近）
                sorted_results = sorted(zip(indices[0], distances[0]), key=lambda x: x[1], reverse=False)
                # 每个维度取对应数量，且去重
                for idx, dist in sorted_results:
                    # 去重，同一句子只出现一次
                    if idx not in choosed_idxs.keys() and n_dims[i] > 0 and self.sents[idx] != query:
                        choosed_idxs[idx] = dist
                        n_dims[i] -= 1
                        
                        if verbose:
                            print(f'{feture_types[i]}: {self.sents[idx]}')
                    if n_dims[i] == 0:
                        break
            # 将字典转换为列表，然后按照距离排序，距离大的放到前面，距离小的放到后面（离输入更近）
            choosed_idxs = sorted(choosed_idxs.items(), key=lambda x: x[1], reverse=True)
            res = [(self.sents[idx], self.data_dict[self.sents[idx]]) for idx, dist in choosed_idxs]
            return res
        else:
            query_embedding = self.encode_sentences([query])
            distances, indices = self.index.search(query_embedding, self.index.ntotal)

            # 距离越大的放到前面（这样最相似的例子离输入最近）
            sorted_results = sorted(zip(indices[0], distances[0]), key=lambda x: x[1], reverse=True)
            # Getting the top k results
            top_results = sorted_results[:selected_k]
            res = [(self.sents[idx], self.data_dict[self.sents[idx]]) for idx, dist in top_results]
            return res


if __name__ == '__main__':
    ex_file_path = "/root/CMDAG_Qwen/Fine_tuning_data/Fine_tuning_train.json"
    paths = {
        'gnn_path': '/root/CMDAG_Qwen/gnn_checkpoints/best_gnn_model.pt',
        'bert_path': '/hy-tmp/models/bert-base-chinese'
    }

    retriever = Ex_Retriver(ex_file_path,paths=paths, encode_method='gnn')

    res = retriever.search_examples("科技就像是一把双刃剑，它既带来了便利和进步，也带来了潜在的威胁和灾难。", selected_k=3)

    examples_str = ""
    for id,example in enumerate(res):
        examples_str += f'示例 {id + 1}:\n输入："{example[0]}"\n输出："{example[1]}"\n'

    print(examples_str)
