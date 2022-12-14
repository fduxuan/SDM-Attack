# -*- coding: utf-8 -*-
'''
Created on: 2022-12-08 09:44:29
LastEditTime: 2022-12-13 13:05:11
Author: fduxuan

Desc:  

'''
import torch
from torch import nn
import modules
import dataloader
from torch.autograd import Variable
from util import *
from pydantic import BaseModel
from typing import List
from transformers import WEIGHTS_NAME, CONFIG_NAME
from transformers import AutoTokenizer, AutoModel, AdamW, AutoModelForSequenceClassification

torch.cuda.set_device(0)

NET_CONFIG='net.pkl'

class ClassificationItem(BaseModel):
    text: str = ""
    words: List[str] = []
    label: int = 0

class VictimModel(nn.Module):
    
    def __init__(self, embedding, hidden_size=150, depth=1, dropout=0.3, cnn=False, nclasses=2):
        super(VictimModel, self).__init__()
        self.cnn = cnn
        self.drop = nn.Dropout(dropout)
        self.emb_layer = modules.EmbeddingLayer(
            embs = dataloader.load_embedding(embedding)
        )
        self.word2id = self.emb_layer.word2id
        
        if cnn:
            self.encoder = modules.CNN_Text(
                self.emb_layer.n_d,
                widths = [3,4,5],
                filters=100
            )
            d_out = 3*100
        else:
            self.encoder = nn.LSTM(
                self.emb_layer.n_d,
                hidden_size//2,
                depth,
                dropout = dropout,
                # batch_first=True,
                bidirectional=True
            )
            d_out = hidden_size
        self.out = nn.Linear(d_out, nclasses)
        self.to('cuda')
        
    def forward(self, input):
        if self.cnn:
            input = input.t()
        emb = self.emb_layer(input)
        emb = self.drop(emb)

        if self.cnn:
            output = self.encoder(emb)
        else:
            output, hidden = self.encoder(emb)
            # output = output[-1]
            output = torch.max(output, dim=0)[0].squeeze()

        output = self.drop(output)
        return self.out(output)

    def text_pred(self, texts: list, batch_size=32):
        batches_x = dataloader.create_batches_x(
            texts,
            batch_size, ##TODO
            self.word2id
        )
        outs = []
        with torch.no_grad():
            for x in batches_x:
                x = Variable(x)
                if self.cnn:
                    x = x.t()
                emb = self.emb_layer(x)

                if self.cnn:
                    output = self.encoder(emb)
                else:
                    output, hidden = self.encoder(emb)
                    # output = output[-1]
                    output = torch.max(output, dim=0)[0]

                outs.append(torch.nn.functional.softmax(self.out(output), dim=-1))

        return torch.cat(outs, dim=0)


class VictimModelForTransformer:
    
    def __init__(self, checkpoint, num_labels=2) -> None:
        self.device = 'cuda'
        self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=num_labels).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    
    def text_pred(self, texts: list, batch_size=32):
        self.model.eval()
        data = [" ".join(x) for x in texts]
        probabilities = []
        num = len(texts) // batch_size
        if len(texts) % batch_size != 0:
            num += 1
        for i in range(0, num):
            
            encode_data = self.tokenizer(
                text=data[i*batch_size: (i+1)*batch_size],
                return_tensors='pt', max_length=512, truncation=True, padding=True
            ).to(self.device)
            logits = self.model(**encode_data).logits
            probability = torch.nn.functional.softmax(logits, dim=-1).cpu()
            probabilities.append(probability)
        return torch.cat(probabilities, dim=0)

    def eval(self):
        self.model.eval()
        
class Discriminator(nn.Module):
    """决策器模型

    Args:
        torch (_type_): 实现对序列的多分类 长度为n则分为 n类 最大的那个作为决策删除
    """
    def __init__(self, checkpoint: str = None) -> None:
        super().__init__()
        self.device = 'cuda'
        self.hidden_size = 768
        self.dropout = nn.Dropout(0.2)
        self.model = None
        self.linear = nn.Linear(self.hidden_size*2, 1)  # 长度为n的概率  需要尾部增加相同长度 0, 1 编码
        if checkpoint is None:
            info(f"load model from bert-base-uncased")
            self.model = AutoModel.from_pretrained('bert-base-uncased').to(self.device)
        else:
            info(f"load model from {checkpoint}")
            self.model = AutoModel.from_pretrained(checkpoint).to(self.device)
            state_dict = torch.load(f"{checkpoint}/{NET_CONFIG}")
            self.load_state_dict(state_dict)
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        finish()
        self.to(self.device)
        
    def forward(self, state_infos,  **kwargs):
        output = self.model(**kwargs)
        drop_output = self.dropout(output.last_hidden_state)  # [B, T, 768]
        # 增加 0, 1 值  state_info : [B, T, 1]   last_dim: 0/1
        state = state_infos.expand(state_infos.shape[0], state_infos.shape[1], 768)
        output = torch.cat((drop_output, state), dim=-1)  # 拼接
        logits = self.linear(output)
        return logits.squeeze(-1) # [B, T]
    
    def saveModel(self, checkpoint: str = "checkpoint"):
        import os
        folder = os.path.exists(checkpoint)
        
        info(f"保存模型至 {checkpoint}")
        if not folder:
            os.makedirs(checkpoint)
        info(f"保存PLM...")
        torch.save(self.model.state_dict(), f"{checkpoint}/{WEIGHTS_NAME}")
        self.model.config.to_json_file(f"{checkpoint}/{CONFIG_NAME}")   
        info('\t保存PyTorch网络参数')
        torch.save(self.state_dict(), f"{checkpoint}/{NET_CONFIG}")
        f"{checkpoint}/{WEIGHTS_NAME}"
        finish()    
    
