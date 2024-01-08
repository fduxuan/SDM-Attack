# -*- coding: utf-8 -*-
'''
Created on: 2022-10-26 08:51:46
LastEditTime: 2023-01-04 08:04:48
Author: fduxuan

Desc:  使用 Universal Sentence Encoder  计算相似度  conda activate tf

'''
import tensorflow_hub as hub
import tensorflow.compat.v1 as tf
import numpy as np
import tqdm
import json
import argparse
from util import *
from model import *
import numpy as np

tf.disable_v2_behavior()
# os.environ['CUDA_VISIBLE_DEVICES'] = " "

class USE:
    
    def __init__(self, result_path, USE_path, task) -> None:
        self.result_path = result_path
        self.USE_path = USE_path
        self.task = task
        
        info('Start load USE')
        self.embed = hub.load(self.USE_path)
        finish()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # cpu_config = tf.ConfigProto(intra_op_parallelism_threads = 8, inter_op_parallelism_threads = 8, device_count = {'CPU': 8})
        # config.gpu_options.per_process_gpu_memory_fraction = 0.1
        self.sess = tf.Session(config=config)
        self.build_graph()
        self.sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
 
    def build_graph(self):
        tf.disable_eager_execution()
        self.sts_input1 = tf.placeholder(tf.string, shape=(None))
        self.sts_input2 = tf.placeholder(tf.string, shape=(None))

        sts_encode1 = tf.nn.l2_normalize(self.embed(self.sts_input1), axis=1)
        sts_encode2 = tf.nn.l2_normalize(self.embed(self.sts_input2), axis=1)
        self.cosine_similarities = tf.reduce_sum(tf.multiply(sts_encode1, sts_encode2), axis=1)
        clip_cosine_similarities = tf.clip_by_value(self.cosine_similarities, -1.0, 1.0)
        self.sim_scores = 1.0 - tf.acos(clip_cosine_similarities)

    def semantic_sim(self, sents1, sents2):
        scores = self.sess.run(
            [self.sim_scores],
            feed_dict={
                self.sts_input1: sents1,
                self.sts_input2: sents2,
            })
        return scores 

    def run(self):
     
        acc = 0
        org_acc = 0
        perturb = []
        change_length = 0
        words_length = 0
        count = 0
        data = json.load(open(self.result_path, 'r'))
        bar = tqdm.tqdm(data)
        attack_rate = 0
        threshold = 0.5
        sim = []
        proportion = 0 # 大于 threshold的attack rate
        for i, d in enumerate(bar):
            perturb2 = d.get('perturb', 0)
            if perturb2 > 0.4:
                d['success'] = 1
            # imdb 0.1
            
            
            acc += d['success'] == 1
            org_acc += d['success']!=0
            attack_rate += d['success'] == 2
            # if d['success'] == 2: # 统计攻击成功的
            if d['success']==2:
                corr = self.semantic_sim([d['org_seq']], [d['adv_seq']])[0].item()
                sim.append(corr)
                # corr = self.semantic_sim([d['seq_a']], [d['adv']])[0]
                if d['success'] == 2 and corr > threshold:
                    proportion += 1
                perturb.append(len(d['change'])/len(d['org_seq'].split()))
                
                
                count += 1
                bar.set_postfix({'sim': np.mean(sim), 'perturb': np.mean(perturb), 'org_acc': org_acc/(i+1)})
        print(f'acc: {acc/len(data)}', f'sim: {np.mean(sim)}', f"perturb: {np.mean(perturb)}", f"attack_rate: {count/len(data)}", f"proportion: {proportion/len(data)}", f"org_acc: {org_acc/len(data)}")
 



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", 
                        type=str, 
                        required=True, 
                        help="The result json file to be evaluated") 
    parser.add_argument("--USE_path", 
                        type=str, 
                        default = "./data/USE_Model",
                        help="Path to the USE encoder.")
    parser.add_argument("--task", 
                        type=str, 
                        choices=["classification", "nli"],
                        default='classification',
                        help="NLP task, default to classification")
    args = parser.parse_args()
    u = USE(args.result_path, args.USE_path, args.task)
    u.run()

if __name__ == "__main__":
    main()