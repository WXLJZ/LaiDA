import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertModel
import re
import json
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')

from gnnencoder.models import MultiHeadGAT

nhead = 1
token_dim = 768
hidden_dim = 128
output_dim = 512

class SentenceEncoder:
    def __init__(self, gnn_model_path, bert_model_path, max_len=256):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # 初始化GNN模型并加载权重
        self.gnn_model = MultiHeadGAT(nhead, token_dim, hidden_dim, output_dim).to(self.device)
        # if torch.cuda.device_count() > 1:
        #     self.gnn_model = nn.DataParallel(self.gnn_model, device_ids=[i for i in range(torch.cuda.device_count())])
        self.gnn_model.load_state_dict(torch.load(gnn_model_path))
        self.gnn_model.eval()
        
        # 初始化BERT模型用于编码句子
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_path)
        self.bert_model = BertModel.from_pretrained(bert_model_path).to(self.device)
        # if torch.cuda.device_count() > 1:
        #     self.bert_model = nn.DataParallel(self.bert_model)
        self.bert_model.eval()
        
        self.max_len = max_len

    def encode(self, sentences):
        # 1. 使用BERT模型编码句子
        encoded_sentences = self._batch_encode_nodes(sentences)
        # 2. 使用GNN模型进一步编码得到句子表征
        with torch.no_grad():
            dims_representations, avg_representation = self.gnn_model(encoded_sentences)
        return dims_representations, avg_representation

    def _batch_encode_nodes(self, inputs):
        """
        批量编码文本为节点特征
        """
        encoded_inputs = self.tokenizer.batch_encode_plus(inputs, return_tensors='pt', padding='max_length',
                                                          max_length=self.max_len, truncation=True)
        input_ids = encoded_inputs['input_ids'].to(self.device)
        attention_mask = encoded_inputs['attention_mask'].to(self.device)  # 获取attention_mask

        with torch.no_grad():
            embeddings = self.bert_model(input_ids, attention_mask=attention_mask).last_hidden_state  # 使用attention_mask
        return embeddings