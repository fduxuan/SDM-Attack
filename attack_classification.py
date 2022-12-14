# -*- coding: utf-8 -*-
'''
Created on: 2022-12-08 06:53:08
LastEditTime: 2022-12-14 04:58:19
Author: fduxuan

Desc:    export HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
Refer to TextFooler
'''
import dataloader
import torch
import argparse
import os
from util import *
import tqdm
from model import *
import numpy as np
import random
import string
from datasets import load_dataset
from USE import *
import nltk


class AttackClassification:
    
    def __init__(self, 
                 dataset_path, 
                 target_model_mode,
                 victim: VictimModel,
                 discriminator: Discriminator,
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
        self.target_model_mode = target_model_mode
        self.victim = victim
        self.discriminator = discriminator
        
        self.idx2word = {}
        self.word2idx = {}
        self.counter_fitting_embeddings_path = counter_fitting_embeddings_path
        self.counter_fitting_cos_sim_path = counter_fitting_cos_sim_path
        self.cos_sim = None
        self.stop_words = stop_words
        self.build_vocab()
        
        # 其他参数
        self.output_file = f"{output_dir}/{output_json}"
        self.sim_score_threshold = sim_score_threshold
        self.synonym_num = synonym_num
        self.perturb_ratio = perturb_ratio
        self.checkpoint = discriminator_checkpoint
        self.mode = mode

        self.random = False
        
        self.USE = USE("d", "./data/USE_Model", "dd")
        
        
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
        finish() 

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
    
    def check_pos(self, org_text):
        word_n_pos_list = nltk.pos_tag(org_text, tagset="universal")
        _, pos_list = zip(*word_n_pos_list)
        return pos_list

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
        encode_data = {'input_ids': torch.tensor([input_ids]).to(self.discriminator.device)}
        state_infos = torch.tensor(state).unsqueeze(dim=0).unsqueeze(dim=-1).to(self.discriminator.device)
        
        logits = self.discriminator(state_infos, **encode_data)
        return state, word2token, logits

    def replace(self, origin_words, origin_label, words: list, index, synonyms: dict, org_pos):
        word = words[index]
        syn = synonyms.get(word, [])
        org_text = words[1: -1]
        org_pos = self.check_pos(org_text)
        threshold = 0.1
        last_prob = self.victim.text_pred([words]).cpu()[0][origin_label]
        if len(org_text) < 15:
            threshold = 0.1
            
        batch_data = [] #(words, syn, sim)
        for s in syn:
            item = words[1: index] + [s[0]] + words[index+1: -1]
            adv_pos = self.check_pos(item)
            # adv_pos = org_pos
            if adv_pos[index-1] == org_pos[index-1] or (set([adv_pos[index-1], org_pos[index-1]])<= set(['NOUN', 'VERB'])):
                # sim_score = self.USE.semantic_sim([" ".join(org_text)], [" ".join(item)])[0].item()
                sim_score = 1
                if sim_score > threshold:  
                    # if True:
                    batch_data.append((item, s, sim_score))
                
        
        if len(batch_data) == 0:
            return None, None
        
        probability = self.victim.text_pred([x[0] for x in batch_data], batch_size=50)
        
        scores = probability[:, origin_label]
        _, slices = scores.sort()
        
        if scores[slices[0]] < 0.5:
            a = []
            for x in slices:
                if scores[x] < 0.5:
                    a.append([batch_data[x][0], probability[x], self.USE.semantic_sim([" ".join(org_text)], [" ".join(batch_data[x][0])])[0].item()])
                else:
                    break
            a = sorted(a, key=lambda x: x[2], reverse=True)
            return a[0][0], a[0][1]
        # else:
        #     flag = scores[slices[0]]
        #     a = []
        #     for x in slices:
        #         if scores[x] - flag <= 0.001:
        #             a.append([batch_data[x][0], probability[x], batch_data[x][2]])
        #         else:
        #             break
        #     a = sorted(a, key=lambda x: x[2], reverse=True) 
        #     return a[0][0], a[0][1] 
        if scores[slices[0]]  >= last_prob:
            return None, None   
        return batch_data[slices[0]][0], probability[slices[0]]  # 在对应标签上最低的
    
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
            # if self.mode == 'eval':
            if True:
                _, slices = torch.tensor(token_index_p).sort(descending=True)  # 降序取最大
                token_index = token_index_list[slices[0]]
            # else:
            #     # token_index_p = np.array(torch.nn.functional.softmax(torch.tensor(token_index_p), dim=-1))
            #     sub = 1 - sum(token_index_p)
            #     token_index_p = [x + sub/len(token_index_p) for x in token_index_p]
            #     token_index = np.random.choice(token_index_list, p=np.array(token_index_p).ravel())
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
        orig_prob = self.victim.text_pred([item.words])[0]
        orig_label = torch.argmax(orig_prob, dim=-1)
        res = {'success': 0, 'org_label': item.label, 'org_seq': org_text, 'adv_seq': org_text, 'change': []}
        if item.label != orig_label:  # 本身不正确，不进行attack
            return res
        res['success'] = 1  
        # 初始化环境
        pool = set()
        # org_pos = self.check_pos(words)
        org_pos = None
        words = ['[CLS]']+ item.words +['[SEP]']

        
        # org_pos = None
        synonyms = self.find_synonyms(words, k=self.synonym_num, threshold=self.sim_score_threshold)
        for key, w in enumerate(words):
            if w in self.stop_words or w in ['[CLS]', '[SEP]'] or w in string.punctuation or w not in synonyms:
                pool.add(key)
        
        origin_pool = pool.copy()
        origin_words = words.copy()
        
        with torch.enable_grad():
            self.discriminator.train()
            self.victim.eval()
            
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
                    
                    orig_prob= self.victim.text_pred([words[1:-1]])[0]
                    state, word2token, logits = self.encode_state(words, pool)
                    
                    word_index, token_index, logits = self.do_discrimination(state, word2token, logits)

                    # 修改状态池
                    pool.add(word_index)
                    syn = self.replace(origin_words, orig_label, words, word_index, synonyms, org_pos)  #(word, sim_score)
                    # 进入victim model
                    words[word_index] = syn[0]
                    new_text = " ".join(words[1:-1])
                    attack_prob = self.victim.text_pred([words[1:-1]])[0]
                    attack_label = torch.argmax(attack_prob, dim=-1)
                    change.append([word_index, origin_words[word_index], words[word_index]])
                    sub = (orig_prob[orig_label] - attack_prob[orig_label]).item()
                   
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
                if constant == 0:
                    optimizer.step()
                
                perturb = len(change)/len(item.words)
                
                bar.set_postfix({'step': step, "perturb": perturb})
            
        return res


    def attack_eval(self, item: ClassificationItem, num: int = 0):
        """success: 0 --> 本身不正确
                    1 --> perturb超过0.4  attack失败
                    2 --> attack成功

        Args:
            item (ClassificationItem): _description_
        """
        with torch.no_grad():
            org_text = " ".join(item.words)
            orig_prob = self.victim.text_pred([item.words]).cpu()[0]
            orig_label = torch.argmax(orig_prob, dim=-1)
            res = {'success': 0, 'org_label': item.label, 'org_seq': org_text, 'adv_seq': org_text, 'change': []}
            if item.label != orig_label:  # 本身不正确，不进行attack
                return res
            
            res['success'] = 1  
            # 初始化环境
            pool = set()
            # org_pos = self.check_pos(words)
            org_pos = None
            words = ['[CLS]']+ item.words +['[SEP]']
            
            
            synonyms = self.find_synonyms(words, k=self.synonym_num, threshold=self.sim_score_threshold)
            for key, w in enumerate(words):
                if w in self.stop_words or w in ['[CLS]', '[SEP]'] or w in string.punctuation or w not in synonyms:
                    pool.add(key)
        
            change = []
            origin_words = words.copy()
            step = 0
            while len(pool) < len(words):
                encode_state, state, word2token = self.encode_state(words, pool)         
                word_index, token_index, logits = self.do_discrimination(encode_state, state, word2token)
                pool.add(word_index)    
                new_words, attack_prob = self.replace(origin_words, orig_label, words, word_index, synonyms, org_pos)  #(word, sim_score)
                # # 进入victim model
                if new_words is None:
                    # pool.add(word_index)
                    continue
                words = ["[CLS]"] + new_words + ['SEP']
                new_text = " ".join(words[1:-1])
                # attack_prob = self.victim.text_pred([words[1:-1]])[0].cpu()
                attack_label = torch.argmax(attack_prob, dim=-1)
                
                change.append([word_index, origin_words[word_index], words[word_index]])
                step += 1
                perturb = len(change)/len(item.words)
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
            # self.discriminator = Discriminator(self.checkpoint)
            self.discriminator.eval()
            self.victim.eval()
            attack_func = self.attack_eval
        for i, item in enumerate(bar):
            try:
                res = attack_func(item, 50)
            except Exception as e:
                print(f"{e}")
                raise e
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
        info(f'dump {self.output_file}')
        if self.mode == 'train':
            self.discriminator.saveModel(self.checkpoint)
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True, 
                        help="Which dataset to attack.")
    parser.add_argument("--num_labels", type=int, default=2,
                        help="How many classes for classification.")
    parser.add_argument("--target_model_mode", type=str,
                        choices=['transformer', 'CNN', 'LSTM'], default='transformer',
                        help="Victim Models Mode")
    parser.add_argument("--target_model_path", type=str, required=True,
                        help="Victim Models checkpoint")
    parser.add_argument("--word_embeddings_path", type=str,
                        help="ath to the word embeddings for the target model")
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
    
    if args.target_model_mode == 'CNN':
        victim = VictimModel(args.word_embeddings_path, cnn=True)
        checkpoint = torch.load(args.target_model_path, map_location='cuda:0')
        victim.load_state_dict(checkpoint)
    elif args.target_model_mode == 'LSTM':
        victim = VictimModel(args.word_embeddings_path, cnn=False)
        checkpoint = torch.load(args.target_model_path, map_location='cuda:0')
        victim.load_state_dict(checkpoint)
    elif args.target_model_mode == 'transformer':
        victim = VictimModelForTransformer(args.target_model_path, num_labels=args.num_labels)
    
    if args.mode == 'eval':
        discriminator = Discriminator(args.discriminator_checkpoint)
    
    a = AttackClassification(args.dataset_path, 
                  args.target_model_mode,
                  victim,
                  discriminator,           
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
    
    