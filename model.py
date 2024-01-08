# -*- coding: utf-8 -*-
'''
Created on: 2022-12-08 09:44:29
LastEditTime: 2022-12-31 09:10:56
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
from transformers import AutoTokenizer, AutoModel, AdamW, AutoModelForSequenceClassification, BertForSequenceClassification
from esim.model import ESIM
from torch.utils.data import Dataset, DataLoader, SequentialSampler, TensorDataset
from InferSent.models import NLINet

torch.cuda.set_device(0)

NET_CONFIG='net.pkl'

class ClassificationItem(BaseModel):
    text: str = ""
    words: List[str] = []
    label: int = 0
    
    def replace_word_at_index(self, index, word):
        new_words = self.words[:index] + [word] + self.words[index+1:]
        return ClassificationItem(text=" ".join(new_words), words=new_words)
    
class NilItem(BaseModel):
    premises: List[str] = []
    hypotheses: List[str] = []
    # words: List[str] = []
    label: int = 0


class VictimModel:
    def text_pred(self, texts: list, batch_size=32):
        pass

class VictimModelForCNN(nn.Module, VictimModel):
    
    def __init__(self, embedding, hidden_size=150, depth=1, dropout=0.3, cnn=False, nclasses=2):
        super().__init__()
        self.cnn = cnn
        self.drop = nn.Dropout(dropout)
        self.emb_layer = modules.EmbeddingLayer(
            embs = dataloader.load_embedding(embedding)
        )
        self.word2id = self.emb_layer.word2id
        
        if cnn:
            info('cnn load')
            self.encoder = modules.CNN_Text(
                self.emb_layer.n_d,
                widths = [3,4,5],
                filters=100
            )
            d_out = 3*100
        else:
            info('lstm load')
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


class VictimModelForTransformer(VictimModel):
    
    def __init__(self, checkpoint, num_labels=2, task='classification') -> None:
        super(VictimModel, self).__init__()
        self.device = 'cuda'
        self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=num_labels).to(self.device)
        # self.model = BertForSequenceClassification.from_pretrained(checkpoint, num_labels=num_labels).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.task = task
    
    def text_pred(self, texts: list, batch_size=32):
        self.model.eval()
        if self.task == 'classification':
            data = [" ".join(x) for x in texts]
        else:
            data = []
            for x in texts:
                p = (" ".join(x)).split('[SEP]')
                data.append((p[0], p[1]))
        probabilities = []
        num = len(texts) // batch_size
        if len(texts) % batch_size != 0:
            num += 1
        with torch.no_grad():
            for i in range(0, num):
             
                if self.task == 'NLI': # 需要手动拼接
                    encode_data = self.tokenizer(
                    text=data[i*batch_size: (i+1)*batch_size],
                    return_tensors='pt', max_length=512, truncation=True, padding=True
                ).to(self.device)
                else:
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
    
    def train(self):
        self.model.train()
    
    def saveModel(self, checkpoint: str = "checkpoint"):
        import os
        folder = os.path.exists(checkpoint)
        
        info(f"保存模型至 {checkpoint}")
        if not folder:
            os.makedirs(checkpoint)
        info(f"保存PLM...")
        self.tokenizer.save_pretrained(checkpoint)
        self.model.save_pretrained(checkpoint)
        finish()  
        
class NLIDataset_InferSent(Dataset):
    """
    Dataset class for Natural Language Inference datasets.

    The class can be used to read preprocessed datasets where the premises,
    hypotheses and labels have been transformed to unique integer indices
    (this can be done with the 'preprocess_data' script in the 'scripts'
    folder of this repository).
    """

    def __init__(self,
                 embedding_path,
                 data,
                 word_emb_dim=300,
                 batch_size=32,
                 bos="<s>",
                 eos="</s>"):
        """
        Args:
            data: A dictionary containing the preprocessed premises,
                hypotheses and labels of some dataset.
            padding_idx: An integer indicating the index being used for the
                padding token in the preprocessed data. Defaults to 0.
            max_premise_length: An integer indicating the maximum length
                accepted for the sequences in the premises. If set to None,
                the length of the longest premise in 'data' is used.
                Defaults to None.
            max_hypothesis_length: An integer indicating the maximum length
                accepted for the sequences in the hypotheses. If set to None,
                the length of the longest hypothesis in 'data' is used.
                Defaults to None.
        """
        self.bos = bos
        self.eos = eos
        self.word_emb_dim = word_emb_dim
        self.batch_size = batch_size

        # build word dict
        self.word_vec = self.build_vocab(data['premises']+data['hypotheses'], embedding_path)

    def build_vocab(self, sentences, embedding_path):
        word_dict = self.get_word_dict(sentences)
        word_vec = self.get_embedding(word_dict, embedding_path)
        print('Vocab size : {0}'.format(len(word_vec)))
        return word_vec

    def get_word_dict(self, sentences):
        # create vocab of words
        word_dict = {}
        for sent in sentences:
            for word in sent:
                if word not in word_dict:
                    word_dict[word] = ''
        word_dict['<s>'] = ''
        word_dict['</s>'] = ''
        word_dict['<oov>'] = ''
        return word_dict

    def get_embedding(self, word_dict, embedding_path):
        # create word_vec with glove vectors
        word_vec = {}
        word_vec['<oov>'] = np.random.normal(size=(self.word_emb_dim))
        with open(embedding_path) as f:
            for line in f:
                word, vec = line.split(' ', 1)
                if word in word_dict:
                    word_vec[word] = np.array(list(map(float, vec.split())))
        print('Found {0}(/{1}) words with embedding vectors'.format(
            len(word_vec), len(word_dict)))
        return word_vec

    def get_batch(self, batch, word_vec, emb_dim=300):
        # sent in batch in decreasing order of lengths (bsize, max_len, word_dim)
        lengths = np.array([len(x) for x in batch])
        max_len = np.max(lengths)
        #         print(max_len)
        embed = np.zeros((max_len, len(batch), emb_dim))

        for i in range(len(batch)):
            for j in range(len(batch[i])):
                if batch[i][j] in word_vec:
                    embed[j, i, :] = word_vec[batch[i][j]]
                else:
                    embed[j, i, :] = word_vec['<oov>']
        #                     embed[j, i, :] = np.random.normal(size=(emb_dim))

        return torch.from_numpy(embed).float(), lengths

    def transform_text(self, data):
        # transform data into seq of embeddings
        premises = data['premises']
        hypotheses = data['hypotheses']

        # add bos and eos
        premises = [['<s>'] + premise + ['</s>'] for premise in premises]
        hypotheses = [['<s>'] + hypothese + ['</s>'] for hypothese in hypotheses]

        batches = []
        for stidx in range(0, len(premises), self.batch_size):
            # prepare batch
            s1_batch, s1_len = self.get_batch(premises[stidx:stidx + self.batch_size],
                                              self.word_vec, self.word_emb_dim)
            s2_batch, s2_len = self.get_batch(hypotheses[stidx:stidx + self.batch_size],
                                              self.word_vec, self.word_emb_dim)
            batches.append(((s1_batch, s1_len), (s2_batch, s2_len)))

        return batches     
    
           
class VictimModelForInferSent(nn.Module, VictimModel):
    def __init__(self, checkpoint, embedding_path, data):
        super(VictimModelForInferSent, self).__init__()
        self.device = 'cuda'
        config_nli_model = {
            'word_emb_dim': 300,
            'enc_lstm_dim': 2048,
            'n_enc_layers': 1,
            'dpout_model': 0.,
            'dpout_fc': 0.,
            'fc_dim': 512,
            'bsize': 32,
            'n_classes': 3,
            'pool_type': 'max',
            'nonlinear_fc': 1,
            'encoder_type': 'InferSent',
            'use_cuda': True,
            'use_target': False,
            'version': 1,
        }
        info("\t* Building model..")
        self.model = NLINet(config_nli_model).cuda()
        info("Reloading pretrained parameters...")
        self.model.load_state_dict(torch.load(checkpoint, map_location='cuda:0'))
        info('Building vocab and embeddings...')
        self.dataset = NLIDataset_InferSent(embedding_path, data=data, batch_size=32)
        
        
    def text_pred(self, texts, batch_size=32):
        self.model.eval()

        # transform text data into indices and create batches
        data_batches = self.dataset.transform_text(texts)

        # Deactivate autograd for evaluation.
        probs_all = []
        with torch.no_grad():
            for batch in data_batches:
                # Move input and output data to the GPU if one is used.
                (s1_batch, s1_len), (s2_batch, s2_len) = batch
                s1_batch, s2_batch = s1_batch.cuda(), s2_batch.cuda()
                logits = self.model((s1_batch, s1_len), (s2_batch, s2_len))
                probs = nn.functional.softmax(logits, dim=-1)
                probs_all.append(probs)

        return torch.cat(probs_all, dim=0)

class NLIDataset_ESIM(Dataset):
    """
    Dataset class for Natural Language Inference datasets.

    The class can be used to read preprocessed datasets where the premises,
    hypotheses and labels have been transformed to unique integer indices
    (this can be done with the 'preprocess_data' script in the 'scripts'
    folder of this repository).
    """

    def __init__(self,
                 worddict_path,
                 padding_idx=0,
                 bos="_BOS_",
                 eos="_EOS_"):
        """
        Args:
            data: A dictionary containing the preprocessed premises,
                hypotheses and labels of some dataset.
            padding_idx: An integer indicating the index being used for the
                padding token in the preprocessed data. Defaults to 0.
            max_premise_length: An integer indicating the maximum length
                accepted for the sequences in the premises. If set to None,
                the length of the longest premise in 'data' is used.
                Defaults to None.
            max_hypothesis_length: An integer indicating the maximum length
                accepted for the sequences in the hypotheses. If set to None,
                the length of the longest hypothesis in 'data' is used.
                Defaults to None.
        """
        self.bos = bos
        self.eos = eos
        self.padding_idx = padding_idx

        # build word dict
        with open(worddict_path, 'rb') as pkl:
            self.worddict = pickle.load(pkl)

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, index):
        return {
            "premise": self.data["premises"][index],
            "premise_length": min(self.premises_lengths[index],
                                  self.max_premise_length),
            "hypothesis": self.data["hypotheses"][index],
            "hypothesis_length": min(self.hypotheses_lengths[index],
                                     self.max_hypothesis_length)
        }

    def words_to_indices(self, sentence):
        """
        Transform the words in a sentence to their corresponding integer
        indices.

        Args:
            sentence: A list of words that must be transformed to indices.

        Returns:
            A list of indices.
        """
        indices = []
        # Include the beggining of sentence token at the start of the sentence
        # if one is defined.
        if self.bos:
            indices.append(self.worddict["_BOS_"])

        for word in sentence:
            if word in self.worddict:
                index = self.worddict[word]
            else:
                # Words absent from 'worddict' are treated as a special
                # out-of-vocabulary word (OOV).
                index = self.worddict['_OOV_']
            indices.append(index)
        # Add the end of sentence token at the end of the sentence if one
        # is defined.
        if self.eos:
            indices.append(self.worddict["_EOS_"])

        return indices

    def transform_to_indices(self, data):
        """
        Transform the words in the premises and hypotheses of a dataset, as
        well as their associated labels, to integer indices.

        Args:
            data: A dictionary containing lists of premises, hypotheses
                and labels, in the format returned by the 'read_data'
                method of the Preprocessor class.

        Returns:
            A dictionary containing the transformed premises, hypotheses and
            labels.
        """
        transformed_data = {"premises": [],
                            "hypotheses": []}

        for i, premise in enumerate(data['premises']):
            # Ignore sentences that have a label for which no index was
            # defined in 'labeldict'.

            indices = self.words_to_indices(premise)
            transformed_data["premises"].append(indices)

            indices = self.words_to_indices(data["hypotheses"][i])
            transformed_data["hypotheses"].append(indices)

        return transformed_data

    def transform_text(self, data):
        #         # standardize data format
        #         data = defaultdict(list)
        #         for hypothesis in hypotheses:
        #             data['premises'].append(premise)
        #             data['hypotheses'].append(hypothesis)

        # transform data into indices
        data = self.transform_to_indices(data)

        self.premises_lengths = [len(seq) for seq in data["premises"]]
        self.max_premise_length = max(self.premises_lengths)

        self.hypotheses_lengths = [len(seq) for seq in data["hypotheses"]]
        self.max_hypothesis_length = max(self.hypotheses_lengths)

        self.num_sequences = len(data["premises"])

        self.data = {
            "premises": torch.ones((self.num_sequences,
                                    self.max_premise_length),
                                   dtype=torch.long) * self.padding_idx,
            "hypotheses": torch.ones((self.num_sequences,
                                      self.max_hypothesis_length),
                                     dtype=torch.long) * self.padding_idx}

        for i, premise in enumerate(data["premises"]):
            end = min(len(premise), self.max_premise_length)
            self.data["premises"][i][:end] = torch.tensor(premise[:end])

            hypothesis = data["hypotheses"][i]
            end = min(len(hypothesis), self.max_hypothesis_length)
            self.data["hypotheses"][i][:end] = torch.tensor(hypothesis[:end])


class VictimModelForESIM(nn.Module, VictimModel):
    def __init__(self, checkpoint):
        super(VictimModelForESIM, self).__init__()
    
    


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
        

class Discriminator2(nn.Module):
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
        self.linear = nn.Linear(self.hidden_size, 1)  # 长度为n的概率  需要尾部增加相同长度 0, 1 编码
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
        # state = state_infos.expand(state_infos.shape[0], state_infos.shape[1], 768)
        # output = torch.cat((drop_output, state), dim=-1)  # 拼接
        logits = self.linear(drop_output)
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
    
