# -*- coding: utf-8 -*-
'''
Created on: 2022-11-30 17:09:40
LastEditTime: 2022-12-11 14:10:35
Author: fduxuan

Desc:  

'''
from util import *
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModel, AdamW
import numpy as np
from pydantic import BaseModel
import torch
from torch import nn
from datasets import load_dataset
import tqdm
import string
import math
from transformers import WEIGHTS_NAME, CONFIG_NAME
import time
import dataloader
from typing import List
import nltk
import argparse
import os
import random
import copy
# from discrimination import Discriminator


NET_CONFIG='net.pkl'

class NilItem(BaseModel):
    premises: List[str] = []
    hypotheses: List[str] = []
    # words: List[str] = []
    label: int = 0
    
class ClassificationItem(BaseModel):
    text: str = ""
    words: List[str] = []
    label: int = 0

class Discriminator(nn.Module):
    """决策器模型

    Args:
        torch (_type_): 实现对序列的多分类 长度为n则分为 n类 最大的那个作为决策删除
    """
    def __init__(self, checkpoint: str = None) -> None:
        super().__init__()
        self.hidden_size = 768
        self.dropout = nn.Dropout(0.2)
        self.model = None
        self.linear = nn.Linear(self.hidden_size*2, 1)  # 长度为n的概率  需要尾部增加相同长度 0, 1 编码
        if checkpoint is None:
            info(f"load model from bert-base-uncased")
            self.model = AutoModel.from_pretrained('bert-base-uncased').to('cuda')
        else:
            info(f"load model from {checkpoint}")
            self.model = AutoModel.from_pretrained(checkpoint).to('cuda')
            state_dict = torch.load(f"{checkpoint}/{NET_CONFIG}")
            self.load_state_dict(state_dict)
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        finish()
        
        
        self.to('cuda')
        
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
    
class  AttackClassification:
    
    def __init__(self, 
                 dataset_path, 
                 num_labels, 
                 target_model_path, 
                 counter_fitting_embeddings_path,
                 counter_fitting_cos_sim_path,
                 output_dir,
                 output_json,
                 sim_score_threshold,
                 synonym_num,
                 perturb_ratio,
                 discriminator_checkpoint,
                 mode,
                 ) -> None:
        self.dataset_path = dataset_path
        self.plm_id = target_model_path
        self.num_labels = num_labels
        
        self.plm = AutoModelForSequenceClassification.from_pretrained(self.plm_id, num_labels=self.num_labels).to('cuda')
        self.tokenizer = AutoTokenizer.from_pretrained(self.plm_id)
        info("Model built!")
        
        # prepare synonym extractor
        # build dictionary via the embedding file
        self.idx2word = {}
        self.word2idx = {}
        self.counter_fitting_embeddings_path = counter_fitting_embeddings_path
        self.counter_fitting_cos_sim_path = counter_fitting_cos_sim_path
        self.cos_sim = None
        self.build_vocab()
        
        self.stop_words = None
        self.get_stopwords()
        
        self.discriminator = Discriminator()
        
        # 其他参数
        self.output_file = f"{output_dir}/{output_json}"
        self.sim_score_threshold = sim_score_threshold
        self.synonym_num = synonym_num
        self.perturb_ratio = perturb_ratio
        self.checkpoint = discriminator_checkpoint
        self.mode = mode
     
        
        self.random = False
    
    def build_vocab(self):
        info('Building vocab...')
        with open(self.counter_fitting_embeddings_path, 'r') as f:
            for line in f:
                word = line.split()[0]
                if word not in self.idx2word:
                    self.idx2word[len(self.idx2word)] = word
                    self.word2idx[word] = len(self.idx2word) - 1

        info('Building cos sim matrix...')
        self.cos_sim = np.load(self.counter_fitting_cos_sim_path)
        info("Cos sim import finished!")    
        
    def read_data(self, length=368) -> List[ClassificationItem]:
        if self.mode == 'eval':
        # if True:
            texts, labels = dataloader.read_corpus(self.dataset_path)
            data = list(zip(texts, labels))
            data = [ClassificationItem(words=x[0][:length], label=int(x[1]))for x in data]
        elif self.mode == 'train':
            dataset = load_dataset(self.dataset_path)
            train = list(dataset['train'])[:5000]
            data = [ClassificationItem(words=x['text'].lower().split()[:length], label=int(x['label']))for x in train]
        info("Data import finished!")
        return data

        # Function 0: List of stop words
    def get_stopwords(self):
        '''
        :return: a set of 266 stop words from nltk. eg. {'someone', 'anyhow', 'almost', 'none', 'mostly', 'around', 'being', 'fifteen', 'moreover', 'whoever', 'further', 'not', 'side', 'keep', 'does', 'regarding', 'until', 'across', 'during', 'nothing', 'of', 'we', 'eleven', 'say', 'between', 'upon', 'whole', 'in', 'nowhere', 'show', 'forty', 'hers', 'may', 'who', 'onto', 'amount', 'you', 'yours', 'his', 'than', 'it', 'last', 'up', 'ca', 'should', 'hereafter', 'others', 'would', 'an', 'all', 'if', 'otherwise', 'somehow', 'due', 'my', 'as', 'since', 'they', 'therein', 'together', 'hereupon', 'go', 'throughout', 'well', 'first', 'thence', 'yet', 'were', 'neither', 'too', 'whether', 'call', 'a', 'without', 'anyway', 'me', 'made', 'the', 'whom', 'but', 'and', 'nor', 'although', 'nine', 'whose', 'becomes', 'everywhere', 'front', 'thereby', 'both', 'will', 'move', 'every', 'whence', 'used', 'therefore', 'anyone', 'into', 'meanwhile', 'perhaps', 'became', 'same', 'something', 'very', 'where', 'besides', 'own', 'whereby', 'whither', 'quite', 'wherever', 'why', 'latter', 'down', 'she', 'sometimes', 'about', 'sometime', 'eight', 'ever', 'towards', 'however', 'noone', 'three', 'top', 'can', 'or', 'did', 'seemed', 'that', 'because', 'please', 'whereafter', 'mine', 'one', 'us', 'within', 'themselves', 'only', 'must', 'whereas', 'namely', 'really', 'yourselves', 'against', 'thus', 'thru', 'over', 'some', 'four', 'her', 'just', 'two', 'whenever', 'seeming', 'five', 'him', 'using', 'while', 'already', 'alone', 'been', 'done', 'is', 'our', 'rather', 'afterwards', 'for', 'back', 'third', 'himself', 'put', 'there', 'under', 'hereby', 'among', 'anywhere', 'at', 'twelve', 'was', 'more', 'doing', 'become', 'name', 'see', 'cannot', 'once', 'thereafter', 'ours', 'part', 'below', 'various', 'next', 'herein', 'also', 'above', 'beside', 'another', 'had', 'has', 'to', 'could', 'least', 'though', 'your', 'ten', 'many', 'other', 'from', 'get', 'which', 'with', 'latterly', 'now', 'never', 'most', 'so', 'yourself', 'amongst', 'whatever', 'whereupon', 'their', 'serious', 'make', 'seem', 'often', 'on', 'seems', 'any', 'hence', 'herself', 'myself', 'be', 'either', 'somewhere', 'before', 'twenty', 'here', 'beyond', 'this', 'else', 'nevertheless', 'its', 'he', 'except', 'when', 'again', 'thereupon', 'after', 'through', 'ourselves', 'along', 'former', 'give', 'enough', 'them', 'behind', 'itself', 'wherein', 'always', 'such', 'several', 'these', 'everyone', 'toward', 'have', 'nobody', 'elsewhere', 'empty', 'few', 'six', 'formerly', 'do', 'no', 'then', 'unless', 'what', 'how', 'even', 'i', 'indeed', 'still', 'might', 'off', 'those', 'via', 'fifty', 'each', 'out', 'less', 're', 'take', 'by', 'hundred', 'much', 'anything', 'becoming', 'am', 'everything', 'per', 'full', 'sixty', 'are', 'bottom', 'beforehand'}
        '''
        stop_words = ['a', 'about', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'ain', 'all', 'almost', 'alone', 'along', 'already', 'also', 'although', 'am', 'among', 'amongst', 'an', 'and', 'another', 'any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere', 'are', 'aren', "aren't", 'around', 'as', 'at', 'back', 'been', 'before', 'beforehand', 'behind', 'being', 'below', 'beside', 'besides', 'between', 'beyond', 'both',  'but', 'by', 'can', 'cannot', 'could', 'couldn', "couldn't", 'd', 'didn', "didn't", 'doesn', "doesn't", 'don', "don't", 'down', 'due', 'during', 'either', 'else', 'elsewhere', 'empty', 'enough', 'even', 'ever', 'everyone', 'everything', 'everywhere', 'except',  'first', 'for', 'former', 'formerly', 'from', 'hadn', "hadn't",  'hasn', "hasn't",  'haven', "haven't", 'he', 'hence', 'her', 'here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'however', 'hundred', 'i', 'if', 'in', 'indeed', 'into', 'is', 'isn', "isn't", 'it', "it's", 'its', 'itself', 'just', 'latter', 'latterly', 'least', 'll', 'may', 'me', 'meanwhile', 'mightn', "mightn't", 'mine', 'more', 'moreover', 'most', 'mostly',  'must', 'mustn', "mustn't", 'my', 'myself', 'namely', 'needn', "needn't", 'neither', 'never', 'nevertheless', 'next', 'no', 'nobody', 'none', 'noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'o', 'of', 'off', 'on', 'once', 'one', 'only', 'onto', 'or', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'per', 'please','s', 'same', 'shan', "shan't", 'she', "she's", "should've", 'shouldn', "shouldn't", 'somehow', 'something', 'sometime', 'somewhere', 'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore', 'therein', 'thereupon', 'these', 'they','this', 'those', 'through', 'throughout', 'thru', 'thus', 'to', 'too','toward', 'towards', 'under', 'unless', 'until', 'up', 'upon', 'used',  've', 'was', 'wasn', "wasn't", 'we',  'were', 'weren', "weren't", 'what', 'whatever', 'when', 'whence', 'whenever', 'where', 'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while', 'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'with', 'within', 'without', 'won', "won't", 'would', 'wouldn', "wouldn't", 'y', 'yet', 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself', 'yourselves']
        self.stop_words = set(stop_words)      

    def predictor(self, texts: List[str]):
        self.plm.eval()
        encode_data = self.tokenizer(
            text=texts,
            return_tensors='pt', max_length=512, truncation=True, padding=True
        ).to('cuda')
        logits = self.plm(**encode_data).logits
        probability = torch.nn.functional.softmax(logits, dim=-1)
        label = torch.argmax(logits, dim=-1)
        return probability, label
    
    def eval(self):
        """原本acc
        """
        data = self.read_data()
        acc = 0
        bar = tqdm.tqdm(data)
        for i, item in enumerate(bar):
            p, label = self.predictor([item])
            if label[0] == item.label:
                acc += 1
            bar.set_postfix({'acc': acc/(i+1)})
        bar.close()
    
    def find_synonyms(self, words: list, k: int = 50, threshold=0.7):
        """ 从 cos sim 文件中获取同义词
        words: list [w1, w2, w3...]
        k: int  candidate num
        threshold: float  similarity
        return:  {word1: [(syn1, score1), (syn2, score2)]}  score: similarity
        
            > a = AttackClassifier()
            > print(a.find_synonyms(['love', 'good']))
            > {'love': [('adores', 0.9859314), ('adore', 0.97984385), ('loves', 0.96597964), ('loved', 0.915682), ('amour', 0.8290837), ('adored', 0.8212078), ('iike', 0.8113105), ('iove', 0.80925167), ('amore', 0.8067776), ('likes', 0.80531627)], 'good': [('alright', 0.8443118), ('buena', 0.84190375), ('well', 0.8350292), ('nice', 0.78931385), ('decent', 0.7794969), ('bueno', 0.7718264), ('best', 0.7566938), ('guten', 0.7464952), ('bonne', 0.74170655), ('opportune', 0.738624)]}
            
        """    
        ids = [self.word2idx[word] for word in words if word in self.word2idx]
        words = [word for word in words if word in self.word2idx]
        sim_order = np.argsort(-self.cos_sim[ids, :])[:, 1:1+k]
        sim_words = {}  # word: [(syn1, score1, syn2, score2....)]
        
        for idx, word in enumerate(words):
            sim_value = self.cos_sim[ids[idx]][sim_order[idx]]
            mask = sim_value >= threshold
            sim_word, sim_value = sim_order[idx][mask], sim_value[mask]
            sim_word = [self.idx2word[id] for id in sim_word]
            if len(sim_word):
                sim_words[word] = [(sim_word[i], sim_value[i]) for i in range(len(sim_word))]
        return sim_words
    
    def encode_state(self, words, pool):
        input_ids = []
        state = []
        word2token = {}  # index --> index
        for i, w in enumerate(words):
            ids = self.discriminator.tokenizer(w, add_special_tokens=False)['input_ids']
            state += [0] * len(ids) if i in pool else [1]*len(ids)
            word2token[i] = (len(input_ids), len(input_ids) + len(ids))  # 左闭右开
            input_ids += ids
        encode_data = {'input_ids': torch.tensor([input_ids]).to('cuda')}
        # state_infos = torch.tensor(state).unsqueeze(dim=0).unsqueeze(dim=-1)
        # last_hidden_state = self.plm(**encode_data, output_hidden_states=True).hidden_states[-1]
        # state_infos = state_infos.expand(state_infos.shape[0], state_infos.shape[1], 768)
        # output = torch.cat((last_hidden_state.to('cpu'), state_infos), dim=-1)  # 拼接
        state_infos = torch.tensor(state).unsqueeze(dim=0).unsqueeze(dim=-1).to('cuda')
        # print(len(input_ids))
        logits = self.discriminator(state_infos, **encode_data)
        return state, word2token, logits
    
    def replace(self, origin_label, words: list, index, synonyms: dict, org_pos):
        word = words[index]
        syn = synonyms.get(word, [])
        texts = []
        syn_copy = []
      
        for s in syn:
            
            item = words[max(1, index-50): index] + [s[0]] + words[index+1: min(len(words)-1, index+50)]
            # item = words[1: index] + [s[0]] + words[index+1: -1]
            # adv_pos = self.check_pos(item)
            adv_pos = org_pos
            if adv_pos[index] == org_pos[index] or (set([adv_pos[index], org_pos[index]])<= set(['NOUN', 'VERB'])):
                
                text = " ".join(item)
                texts.append(text)
                syn_copy.append(s)
        if len(texts) == 0:
            return None
        probability = None
        # batch: 8

        for i in range(0, len(texts), 16):
            # print(texts[i*8: i*8 + 8])
   
            p, _ = self.predictor(texts[i: i + 16])
            if i == 0:
                probability = p
            else:
                probability = torch.cat((probability, p), 0)
        
        # probability, label = self.predictor_by_words(texts)
        
        scores = probability[:, origin_label]
        _, slices = scores.sort()
        return syn_copy[slices[0]]  # 在对应标签上最低的
    
    def do_discrimination(self, state, word2token, logits):
        """进行一次判别
        根据pool生成state_info  
        Args:
            words (_type_): _description_
            pool (_type_): _description_
            返回要修改的word index
        """
        
        token_index_list = []
        token_index_p = []
        # 寻找state 为 1 里面 logits最大的 
        token_index = 0
        pro = torch.nn.functional.softmax(logits, dim=-1)
        # _, slices = pro.sort()
        for  t, v in enumerate(pro[0]):
            if state[t] == 0:
                continue
            else:
                token_index_list.append(t)
                token_index_p.append(v.item())
                
        if self.random:
            token_index = random.choice(token_index_list)
        else:
            _, slices = torch.tensor(token_index_p).sort(descending=True)  # 降序取最大
            token_index = token_index_list[slices[0]]
        # if self.mode == 'eval':
        # if True:
        #     _, slices = torch.tensor(token_index_p).sort(descending=True)  # 降序取最大
        #     token_index = token_index_list[slices[0]]
        #     # token_index = random.choice(token_index_list)
        # else:
        #     if self.random: # 没找到，则随机找
        #         token_index = random.choice(token_index_list)
        #     else:
        #         token_index_p = np.array(torch.nn.functional.softmax(torch.tensor(token_index_p), dim=-1))
        #         # sub = 1 - sum(token_index_p)
        #         # token_index_p = [x + sub/len(token_index_p) for x in token_index_p]
        #         token_index = np.random.choice(token_index_list, p=np.array(token_index_p).ravel())
        
                   
        word_index = 0
        for i in word2token:
            if word2token[i][0]<= token_index and word2token[i][1] > token_index:
                word_index = i
        return word_index, word2token[word_index][0], logits    
    
    def attack(self, item: ClassificationItem, attempts: int = 50):
        """success: 0 --> 本身不正确
                    1 --> perturb超过0.4  attack失败
                    2 --> attack成功
        episode: 50
        Args:
            item (NilItem): _description_
        """
        org_text = " ".join(item.words)
        orig_probs, orig_labels = self.predictor([org_text])
        orig_label = orig_labels[0]
        res = {'success': 0, 'org_label': item.label, 'org_seq': org_text, 'adv_seq': org_text, 'change': []}
        
        if item.label != orig_label:  # 本身不正确，不进行attack
            return res
        
        res['success'] = 1  
        
        # 初始化环境
        pool = set()
        words = ['[CLS]']+ item.words +['[SEP]']
        org_pos = self.check_pos(words)
        synonyms = self.find_synonyms(words, k=self.synonym_num, threshold=self.sim_score_threshold)
        for key, w in enumerate(words):
            if w in self.stop_words or w in ['[CLS]', '[SEP]'] or w in string.punctuation or w not in synonyms:
                pool.add(key)
        
        origin_pool = pool.copy()
        origin_words = words.copy()
        
        with torch.enable_grad():
            self.discriminator.train()
            
            for parameter in self.plm.parameters():
                parameter.require_grad = False # 冻结PLm
           
            optimizer = AdamW(self.discriminator.parameters(), lr=3e-6)
            bar = tqdm.trange(attempts)
            min_step = 100
            constant = 0
           
            for _ in bar:
                optimizer.zero_grad()
                step = 0
                words = origin_words.copy()
                
                pool = origin_pool.copy()
                change = []
                loss = 0.0
                
                if constant > 25:
                    break
                while len(pool) < len(words):
                    
                    orig_probs, orig_labels = self.predictor([" ".join(words[1:-1])])
                    state, word2token, logits = self.encode_state(words, pool)
                    
                    word_index, token_index, logits = self.do_discrimination(state, word2token, logits)

                    # 修改状态池
                    pool.add(word_index)
                    syn = self.replace(orig_label, words, word_index, synonyms, org_pos)  #(word, sim_score)
                    # 进入victim model
                    words[word_index] = syn[0]
                    new_text = " ".join(words[1:-1])
                    attack_probs, attack_labels = self.predictor([new_text])
                    attack_label = attack_labels[0]
                    change.append([word_index, origin_words[word_index], words[word_index]])
                    sub = (orig_probs[0][orig_label] - attack_probs[0][orig_label]).item()
                   
                    reward = sub # 增加下降值
                    
                    
                    # 计算期望
                    pro = torch.nn.functional.softmax(logits, dim=-1)[0]
                    h = -torch.log(pro[token_index])
                    

                    step += 1
                    perturb = len(change)/len(item.words)
        
                    
                    if perturb > self.perturb_ratio:
                        constant += 1
                        self.random = True
                        loss = -torch.abs(h*reward)
                        loss.backward()
                        break
                    
                    loss = h * reward
                    loss.backward()
                    
                    if attack_label != orig_label:
                        self.random = False
                        constant = 0
                        if step <= min_step:
                            min_step = step
                            
                            res['success'] = 2
                            res['adv_label'] = attack_label.item()
                            res['adv_seq'] = new_text
                            res['change'] = change
                            res['perturb'] = perturb
                        break
                optimizer.step()
                
                perturb = len(change)/len(item.words)
                
                bar.set_postfix({'step': step, "perturb": perturb})
            
        return res
    
    def check_pos(self, org_text):
        word_n_pos_list = nltk.pos_tag(org_text, tagset="universal")
        _, pos_list = zip(*word_n_pos_list)
        return pos_list
    
    def attack_eval(self, item: ClassificationItem, num: int = 0):
        """success: 0 --> 本身不正确
                    1 --> perturb超过0.4  attack失败
                    2 --> attack成功

        Args:
            item (ClassificationItem): _description_
        """
        org_text = " ".join(item.words)
        orig_probs, orig_labels = self.predictor([org_text])
        orig_label = orig_labels[0]
        res = {'success': 0, 'org_label': item.label, 'org_seq': org_text, 'adv_seq': org_text, 'change': []}
        if item.label != orig_label:  # 本身不正确，不进行attack
            return res
        
        res['success'] = 1  
        # 初始化环境
        pool = set()
        words = ['[CLS]']+ item.words +['[SEP]']
        org_pos = self.check_pos(words)
        synonyms = self.find_synonyms(words, k=self.synonym_num, threshold=self.sim_score_threshold)
        for key, w in enumerate(words):
            if w in self.stop_words or w in ['[CLS]', '[SEP]'] or w in string.punctuation or w not in synonyms:
                pool.add(key)
    
        change = []
        origin_words = words.copy()
        step = 0
        memory = []
        while len(pool) < len(words):
            encode_state, state, word2token = self.encode_state(words, pool)         
            word_index, token_index, logits = self.do_discrimination(encode_state, state, word2token)
            pool.add(word_index)    
            syn = self.replace(orig_label, words, word_index, synonyms, org_pos)  #(word, sim_score)
            # 进入victim model
            words[word_index] = syn[0]
            new_text = " ".join(words[1:-1])
            attack_probs, attack_labels = self.predictor([new_text])
            attack_label = attack_labels[0]
            change.append([word_index, origin_words[word_index], words[word_index]])
            step += 1
            perturb = len(change)/len(item.words)
            # memory.append([word_index])
            if perturb > self.perturb_ratio:
                break
            if attack_label != orig_label:
                if perturb < self.perturb_ratio:
                    min_step = step
                    flag = 1
                    res['success'] = 2
                    res['adv_label'] = attack_label.item()
                    res['adv_seq'] = new_text
                    res['change'] = change
                    res['perturb'] = perturb
                    break           
        return res
            
        
    def run(self):
        data = self.read_data()
        bar = tqdm.tqdm(data)
        acc = 0
        attack_total = 0
        perturb_total = 0
        perturb = 0.0
        ans = []
        attack_func = self.attack
        # self.discriminator = Discriminator(self.checkpoint)
        if self.mode == 'eval':
            self.discriminator = Discriminator(self.checkpoint)
            self.discriminator.eval()
            attack_func = self.attack_eval
        for i, item in enumerate(bar):
            try:
                res = attack_func(item, 50)
            except Exception as e:
                print(f"{e}")
                # raise e
                res = res = {'success': 1, 'org_label': item.label, 'org_seq': " ".join(item.words), 'adv_seq': " ".join(item.words),'change': []}
            ans.append(res)
            if res['success'] == 1: # 没有攻击成功
                acc += 1
                attack_total += 1
            elif res['success'] == 0:
                pass
            elif res['success'] == 2:
                perturb_total += 1
                perturb += res['perturb']
                attack_total += 1
            if perturb_total and attack_total:
                bar.set_postfix({'acc': acc/(i+1), 'perturb': perturb/perturb_total, 'attack_rate': perturb_total/attack_total})
            if i % 100 == 0 and self.mode == 'train':
                self.discriminator.saveModel(self.checkpoint)    
        import json
        json.dump(ans, open(self.output_file, 'w'))
        print(f"acc: {acc/len(data)}\t perturb: {perturb/perturb_total}\t attack_rate: {perturb_total/attack_total}")
        if self.mode == 'train':
            self.discriminator.saveModel(self.checkpoint)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True, 
                        help="Which dataset to attack.")
    parser.add_argument("--num_labels", type=int, default=2,
                        help="How many classes for classification.")
    parser.add_argument("--target_model_path", type=str, required=True,
                        help="Victim Models")
    parser.add_argument("--counter_fitting_embeddings_path", type=str, required=True,
                        help="path to the counter-fitting embeddings we used to find synonyms")
    parser.add_argument("--counter_fitting_cos_sim_path", type=str, default='',
                        help="pre-compute the cosine similarity scores based on the counter-fitting embeddings")
    parser.add_argument("--USE_path", type=str, 
                        help="Path to the USE encoder.")
    parser.add_argument("--output_dir", type=str, default='adv_results',
                        help="The output directory where the attack results will be written.")
    parser.add_argument("--output_json", type=str, default='result.json',
                        help="The attack results.")
    parser.add_argument("--discriminator_checkpoint", type=str, default='checkpoint',
                        help="The checkpoint directory of agent.")
    parser.add_argument("--mode", type=str, required=True, choices=['train', 'eval'], help='train / eval mode')
    
    ## Model hyperparameters
    parser.add_argument("--sim_score_threshold", default=0.7, type=float,
                        help="Required minimum semantic similarity score.")
    parser.add_argument("--synonym_num", default=50, type=int,
                        help="Number of synonyms to extract")
    parser.add_argument("--perturb_ratio", default=0.4, type=float,
                        help="Whether use random perturbation for ablation study")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="max sequence length for BERT target model")
    
    args = parser.parse_args()
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        print("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    else:
        os.makedirs(args.output_dir, exist_ok=True)
    
    info('Start attacking!' if args.mode == 'eval' else 'Start Training!')
    
    a = AttackClassification(args.dataset_path, 
                  args.num_labels, 
                  args.target_model_path, 
                  args.counter_fitting_embeddings_path,
                  args.counter_fitting_cos_sim_path,
                  args.output_dir,
                  args.output_json,
                  args.sim_score_threshold,
                  args.synonym_num,
                  args.perturb_ratio,
                  args.discriminator_checkpoint,
                  args.mode)
    a.run()
    
if __name__ == "__main__":
    main()