# -*- coding: utf-8 -*-
'''
Created on: 2022-12-20 07:36:29
LastEditTime: 2022-12-23 05:49:09
Author: fduxuan

Desc: 测试可迁移性的success_rate 

'''
from argparse import ArgumentParser
from model import *
import json
from util import *
import tqdm


def main():
    parser = ArgumentParser()
    parser.add_argument('--result_path', type=str, required=True,
                        help='which result file')
    parser.add_argument("--target_model_mode", type=str,
                        choices=['transformer', 'CNN', 'LSTM'], default='transformer',
                        help="Victim Models Mode")
    parser.add_argument("--target_model_path", type=str, required=True,
                        help="Victim Models checkpoint")
    parser.add_argument("--word_embeddings_path", type=str,
                        help="path to the word embeddings for the target model")
    args = parser.parse_args()
    data = json.load(open(args.result_path, 'r'))
    data = [ClassificationItem(words=x['adv_seq'].lower().split(), label=int(x['org_label']))for x in data if x['success']==2]
    info("load data!")
    
    if args.target_model_mode == 'CNN':
        victim = VictimModelForCNN(args.word_embeddings_path, cnn=True)
        checkpoint = torch.load(args.target_model_path, map_location='cuda:0')
        victim.load_state_dict(checkpoint)
    elif args.target_model_mode == 'LSTM':
        victim = VictimModelForCNN(args.word_embeddings_path, cnn=False)
        checkpoint = torch.load(args.target_model_path, map_location='cuda:0')
        victim.load_state_dict(checkpoint)
    elif args.target_model_mode == 'transformer':
        victim = VictimModelForTransformer(args.target_model_path, num_labels=2)
    
    bar = tqdm.tqdm(data)
    attack_rate = 0
    for index, item in enumerate(bar):
        probability = victim.text_pred([item.words])
        label = torch.argmax(probability, dim=-1)[0]
        if label != item.label:
            attack_rate += 1
        bar.set_postfix({'attack_rate': attack_rate/(index+1)})
    
"""
bert -- wordCNN
python3 transfer.py --result_path adv_results/imdb_eval.json --target_model_mode CNN --target_model_path WordCnn/imdb --word_embeddings_path  WordCnn/glove.6B.200d.txt

bert -- wordLSTM
python3 transfer.py --result_path adv_results/imdb_eval.json --target_model_mode LSTM --target_model_path WordLSTM/imdb --word_embeddings_path  WordCnn/glove.6B.200d.txt

wordCNN -- bert
python3 transfer.py --result_path adv_results/WordCNN/imdb_eval.json  --target_model_path textattack/bert-base-uncased-imdb 

wordCNN -- lstm
python3 transfer.py --result_path adv_results/WordCNN/imdb_eval.json --target_model_mode LSTM --target_model_path WordLSTM/imdb --word_embeddings_path  WordCnn/glove.6B.200d.txt

wordLstm --bert
python3 transfer.py --result_path adv_results/WordLSTM/imdb_eval.json  --target_model_path textattack/bert-base-uncased-imdb 

wordLstm --CNN

python3 transfer.py --result_path adv_results/WordLSTM/imdb_eval.json --target_model_mode CNN --target_model_path WordCnn/imdb --word_embeddings_path  WordCnn/glove.6B.200d.txt


"""


"""
imdb 本身
python3 attack_classification.py --mode eval --dataset_path data/imdb  --num_labels 2 --target_model_path textattack/bert-base-uncased-imdb --counter_fitting_embeddings_path data/counter-fitted-vectors.txt --counter_fitting_cos_sim_path data/cos_sim_counter_fitting.npy --output_dir adv_results/transfer --output_json imdb.json --discriminator_checkpoint Bert-Victim/imdb_checkpoint --sim_score_threshold 0.5


imdb -- yelp
python3 attack_classification.py --mode eval --dataset_path data/yelp  --num_labels 2 --target_model_path textattack/bert-base-uncased-yelp-polarity --counter_fitting_embeddings_path data/counter-fitted-vectors.txt --counter_fitting_cos_sim_path data/cos_sim_counter_fitting.npy --output_dir adv_results/transfer --output_json imdb_yelp.json --discriminator_checkpoint Bert-Victim/imdb_checkpoint --sim_score_threshold 0.5

imdb -- mr
python3 attack_classification.py --mode eval --dataset_path data/mr  --num_labels 2 --target_model_path textattack/bert-base-uncased-rotten_tomatoes --counter_fitting_embeddings_path data/counter-fitted-vectors.txt --counter_fitting_cos_sim_path data/cos_sim_counter_fitting.npy --output_dir adv_results/transfer --output_json imdb_mr.json --discriminator_checkpoint Bert-Victim/imdb_checkpoint --sim_score_threshold 0.5
acc: 0.205 sim: 0.6606016218066215 perturb: 0.12361701285059161 org_acc: 0.969 attack_rate: 0.7884416924664602

imdb - ag
python3 attack_classification.py --mode eval --dataset_path data/ag  --num_labels 4 --target_model_path Victim-TextFooler/ag --counter_fitting_embeddings_path data/counter-fitted-vectors.txt --counter_fitting_cos_sim_path data/cos_sim_counter_fitting.npy --output_dir adv_results/transfer --output_json imdb_ag.json --discriminator_checkpoint Bert-Victim/imdb_checkpoint --sim_score_threshold 0.5 --perturb_ratio 0.5

-------------
mr 本身
python3 attack_classification.py --mode eval --dataset_path data/mr  --num_labels 2 --target_model_path textattack/bert-base-uncased-rotten_tomatoes --counter_fitting_embeddings_path data/counter-fitted-vectors.txt --counter_fitting_cos_sim_path data/cos_sim_counter_fitting.npy --output_dir adv_results/transfer --output_json mr.json --discriminator_checkpoint Bert-Victim/mr_checkpoint --sim_score_threshold 0.5
acc: 0.092 sim: 0.6470044802427292 perturb: 0.10014662750561527 org_acc: 0.969 attack_rate: 0.9050567595459237

mr-yelp
python3 attack_classification.py --mode eval --dataset_path data/yelp  --num_labels 2 --target_model_path textattack/bert-base-uncased-yelp-polarity --counter_fitting_embeddings_path data/counter-fitted-vectors.txt --counter_fitting_cos_sim_path data/cos_sim_counter_fitting.npy --output_dir adv_results/transfer --output_json mr_yelp.json --discriminator_checkpoint Bert-Victim/mr_checkpoint --sim_score_threshold 0.5
acc: 0.022 sim: 0.6876309233903884 perturb: 0.08610771220073173 org_acc: 0.972 attack_rate: 0.977366255144033

mr-imdb
python3 attack_classification.py --mode eval --dataset_path data/imdb  --num_labels 2 --target_model_path textattack/bert-base-uncased-imdb --counter_fitting_embeddings_path data/counter-fitted-vectors.txt --counter_fitting_cos_sim_path data/cos_sim_counter_fitting.npy --output_dir adv_results/transfer --output_json mr_imdb.json --discriminator_checkpoint Bert-Victim/mr_checkpoint --sim_score_threshold 0.5

mr-ag
python3 attack_classification.py --mode eval --dataset_path data/ag  --num_labels 4 --target_model_path Victim-TextFooler/ag --counter_fitting_embeddings_path data/counter-fitted-vectors.txt --counter_fitting_cos_sim_path data/cos_sim_counter_fitting.npy --output_dir adv_results/transfer --output_json mr_ag.json --discriminator_checkpoint Bert-Victim/mr_checkpoint --sim_score_threshold 0.5 --perturb_ratio 0.5

-------------
yelp 自身
python3 attack_classification.py --mode eval --dataset_path data/yelp  --num_labels 2 --target_model_path textattack/bert-base-uncased-yelp-polarity --counter_fitting_embeddings_path data/counter-fitted-vectors.txt --counter_fitting_cos_sim_path data/cos_sim_counter_fitting.npy --output_dir adv_results/transfer --output_json yelp.json --discriminator_checkpoint Bert-Victim/yelp_checkpoint --sim_score_threshold 0.5

yelp-imdb
python3 attack_classification.py --mode eval --dataset_path data/imdb  --num_labels 2 --target_model_path textattack/bert-base-uncased-imdb --counter_fitting_embeddings_path data/counter-fitted-vectors.txt --counter_fitting_cos_sim_path data/cos_sim_counter_fitting.npy --output_dir adv_results/transfer --output_json yelp_imdb.json --discriminator_checkpoint Bert-Victim/yelp_checkpoint --sim_score_threshold 0.5

yelp - mr
python3 attack_classification.py --mode eval --dataset_path data/mr  --num_labels 2 --target_model_path textattack/bert-base-uncased-rotten_tomatoes --counter_fitting_embeddings_path data/counter-fitted-vectors.txt --counter_fitting_cos_sim_path data/cos_sim_counter_fitting.npy --output_dir adv_results/transfer --output_json yelp_mr.json --discriminator_checkpoint Bert-Victim/yelp_checkpoint --sim_score_threshold 0.5
acc: 0.135 sim: 0.5539761195778847 perturb: 0.1322616404926577 org_acc: 0.969 attack_rate: 0.8606811145510835

yelp - ag
python3 attack_classification.py --mode eval --dataset_path data/ag  --num_labels 4 --target_model_path Victim-TextFooler/ag --counter_fitting_embeddings_path data/counter-fitted-vectors.txt --counter_fitting_cos_sim_path data/cos_sim_counter_fitting.npy --output_dir adv_results/transfer --output_json yelp_ag.json --discriminator_checkpoint Bert-Victim/yelp_checkpoint --sim_score_threshold 0.5 --perturb_ratio 0.5

-------------
ag 自身
python3 attack_classification.py --mode eval --dataset_path data/ag  --num_labels 4 --target_model_path Victim-TextFooler/ag --counter_fitting_embeddings_path data/counter-fitted-vectors.txt --counter_fitting_cos_sim_path data/cos_sim_counter_fitting.npy --output_dir adv_results/transfer --output_json ag.json --discriminator_checkpoint Bert-Victim/ag_checkpoint --sim_score_threshold 0.5 --perturb_ratio 0.5

ag-imdb
python3 attack_classification.py --mode eval --dataset_path data/imdb  --num_labels 2 --target_model_path textattack/bert-base-uncased-imdb --counter_fitting_embeddings_path data/counter-fitted-vectors.txt --counter_fitting_cos_sim_path data/cos_sim_counter_fitting.npy --output_dir adv_results/transfer --output_json ag_imdb.json --discriminator_checkpoint Bert-Victim/ag_checkpoint --sim_score_threshold 0.5

ag-yelp
python3 attack_classification.py --mode eval --dataset_path data/yelp  --num_labels 2 --target_model_path textattack/bert-base-uncased-yelp-polarity --counter_fitting_embeddings_path data/counter-fitted-vectors.txt --counter_fitting_cos_sim_path data/cos_sim_counter_fitting.npy --output_dir adv_results/transfer --output_json ag_yelp.json --discriminator_checkpoint Bert-Victim/ag_checkpoint --sim_score_threshold 0.5

ag-mr
python3 attack_classification.py --mode eval --dataset_path data/mr  --num_labels 2 --target_model_path textattack/bert-base-uncased-rotten_tomatoes --counter_fitting_embeddings_path data/counter-fitted-vectors.txt --counter_fitting_cos_sim_path data/cos_sim_counter_fitting.npy --output_dir adv_results/transfer --output_json ag_mr.json --discriminator_checkpoint Bert-Victim/ag_checkpoint --sim_score_threshold 0.5

---------------------
random_ag
python3 attack_classification.py --mode eval --dataset_path data/ag  --num_labels 4 --target_model_path Victim-TextFooler/ag --counter_fitting_embeddings_path data/counter-fitted-vectors.txt --counter_fitting_cos_sim_path data/cos_sim_counter_fitting.npy --output_dir adv_results/transfer --output_json random_ag.json --discriminator_checkpoint Bert-Victim/ag_checkpoint --sim_score_threshold 0.5 --perturb_ratio 0.5

random-imdb
python3 attack_classification.py --mode eval --dataset_path data/imdb  --num_labels 2 --target_model_path textattack/bert-base-uncased-imdb --counter_fitting_embeddings_path data/counter-fitted-vectors.txt --counter_fitting_cos_sim_path data/cos_sim_counter_fitting.npy --output_dir adv_results/transfer --output_json random_imdb.json --discriminator_checkpoint Bert-Victim/imdb_checkpoint --sim_score_threshold 0.5

random-yelp
python3 attack_classification.py --mode eval --dataset_path data/yelp  --num_labels 2 --target_model_path textattack/bert-base-uncased-yelp-polarity --counter_fitting_embeddings_path data/counter-fitted-vectors.txt --counter_fitting_cos_sim_path data/cos_sim_counter_fitting.npy --output_dir adv_results/transfer --output_json random_yelp.json --discriminator_checkpoint Bert-Victim/yelp_checkpoint --sim_score_threshold 0.5

random-mr
python3 attack_classification.py --mode eval --dataset_path data/mr  --num_labels 2 --target_model_path textattack/bert-base-uncased-rotten_tomatoes --counter_fitting_embeddings_path data/counter-fitted-vectors.txt --counter_fitting_cos_sim_path data/cos_sim_counter_fitting.npy --output_dir adv_results/transfer --output_json random_mr.json --discriminator_checkpoint Bert-Victim/mr_checkpoint --sim_score_threshold 0.5

"""



if __name__ == '__main__':
    main()