# -*- coding: utf-8 -*-
'''
Created on: 2022-10-26 08:51:46
LastEditTime: 2022-12-14 03:15:39
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
        sim = 0
        acc = 0
        perturb = 0
        count = 0
        data = json.load(open(self.result_path, 'r'))
        bar = tqdm.tqdm(data)
        for d in bar:
            acc += d['success'] == 1
            # if d['success'] == 2: # 统计攻击成功的
            if True:
                corr = self.semantic_sim([d['org_seq']], [d['adv_seq']])[0]
                # corr = self.semantic_sim([d['seq_a']], [d['adv']])[0]
                
                sim += corr
                perturb += d.get('perturb', 0)     
                count += 1
                bar.set_postfix({'sim': sim/count, 'perturb': perturb/count})
        print(f'acc: {acc/len(data)}', f'sim: {sim/count}', f'perturb: {perturb/count}')
 



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