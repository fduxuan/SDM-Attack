# -*- coding: utf-8 -*-
'''
Created on: 2022-12-23 05:53:17
LastEditTime: 2023-01-13 06:03:12
Author: fduxuan

Desc:   adversarial training

'''
from model import *
from argparse import ArgumentParser
import json
import tqdm
from transformers import AdamW
from datasets import load_dataset

class TrainVictim:
    def __init__(self, victim, checkpoint, data) -> None:
        self.victim: VictimModel = victim
        self.checkpoint = checkpoint
        self.data = data
        
    def train(self, mode):
        self.victim.train()
        with torch.enable_grad():
            optimizer = AdamW(self.victim.model.parameters(), lr=3e-5)
            for epoch in range(1):
                bar = tqdm.tqdm(self.data)
                for index, item in enumerate(bar):
                    optimizer.zero_grad()
                    if mode == 'classification':
                        encoding = self.victim.tokenizer([item.text], return_tensors='pt', padding=True, truncation=True).to('cuda')
                    else:
                        t = item.text.split('[SEP]')
                        p = t[0].strip()
                        h = t[1].strip()
                        encoding = self.victim.tokenizer([(p, h)], return_tensors='pt', padding=True, truncation=True).to('cuda')
                        
                    labels = torch.tensor([item.label]).unsqueeze(0).to('cuda')
                    outputs = self.victim.model(**encoding, labels=labels)
                    loss = outputs.loss
                    loss.backward()
                    optimizer.step()
                    bar.set_postfix({'epoch': epoch+1, 'loss': loss.item()})
            self.victim.saveModel(self.checkpoint) 

def main():
    parser = ArgumentParser()
    parser.add_argument('--result_path', type=str, required=True,
                        help='which result file')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='which result file')
    parser.add_argument("--target_model_path", type=str, required=True,
                        help="Victim Models checkpoint")
    parser.add_argument("--num_labels", type=int, default=2,
                        help="num labels")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="save location")
    
    parser.add_argument("--mode", type=str, default="classification",
                        help="classification/nli")
    args = parser.parse_args()

    victim = VictimModelForTransformer(args.target_model_path, num_labels=args.num_labels)
    dataset = json.load(open(args.result_path, 'r'))
    data = []

    for x in dataset[:100]:
        if args.mode == 'classification':
            data.append(ClassificationItem(text=x['adv_seq'].lower(), label=int(x['org_label'])))
            data.append(ClassificationItem(text=x['org_seq'].lower(), label=int(x['org_label'])) )
        else:
            text1 = x['premises'] + " [SEP] " + x['org_seq'] 
            text2 = x['premises'] + " [SEP] " + x['adv_seq'] 
       
            data.append(ClassificationItem(text=text1, label=int(x['org_label'])))
            data.append(ClassificationItem(text=text2, label=int(x['org_label'])))
            
    
    
    info("load data!")
    t = TrainVictim(victim, args.checkpoint,data)
    t.train(args.mode)
    
if __name__ == "__main__":
    main()

"""_summary_
mr 
generate
python3 attack_classification.py --mode eval --dataset_path rotten_tomatoes  --num_labels 2 --target_model_path textattack/bert-base-uncased-rotten_tomatoes --counter_fitting_embeddings_path data/counter-fitted-vectors.txt --counter_fitting_cos_sim_path data/cos_sim_counter_fitting.npy --output_dir adv_results/adv_train --output_json mr.json --discriminator_checkpoint Bert-Victim/mr_checkpoint --sim_score_threshold 0.5
train
python3 train_victim.py --dataset_path rotten_tomatoes --result_path adv_results/mr_eval.json --num_labels 2 --target_model_path textattack/bert-base-uncased-rotten_tomatoes --checkpoint adv_training/mr
eval
python3 attack_classification.py --mode eval --dataset_path data/mr  --num_labels 2 --target_model_path adv_training/mr --counter_fitting_embeddings_path data/counter-fitted-vectors.txt --counter_fitting_cos_sim_path data/cos_sim_counter_fitting.npy --output_dir adv_results/adv_train --output_json mr_eval.json --discriminator_checkpoint Bert-Victim/mr_checkpoint --sim_score_threshold 0.5


ag - train
generate
python3 attack_classification.py --mode eval --dataset_path ag_news  --num_labels 4 --target_model_path Victim-TextFooler/ag --counter_fitting_embeddings_path data/counter-fitted-vectors.txt --counter_fitting_cos_sim_path data/cos_sim_counter_fitting.npy --output_dir adv_results/adv_train --output_json ag.json --discriminator_checkpoint Bert-Victim/ag_checkpoint --sim_score_threshold 0.5 --perturb_ratio 0.5
train
python3 train_victim.py --dataset_path rotten_tomatoes --result_path adv_results/ag_news_eval.json --num_labels 4 --target_model_path Victim-TextFooler/ag --checkpoint adv_training/ag
eval
python3 attack_classification.py --mode eval --dataset_path data/ag --num_labels 4 --target_model_path adv_training/ag --counter_fitting_embeddings_path data/counter-fitted-vectors.txt --counter_fitting_cos_sim_path data/cos_sim_counter_fitting.npy --output_dir adv_results/adv_train --output_json ag_eval.json --discriminator_checkpoint Bert-Victim/mr_checkpoint --sim_score_threshold 0.5 


imdb -train
generate
python3 attack_classification.py --mode eval --dataset_path imdb  --num_labels 2 --target_model_path textattack/bert-base-uncased-imdb --counter_fitting_embeddings_path data/counter-fitted-vectors.txt --counter_fitting_cos_sim_path data/cos_sim_counter_fitting.npy --output_dir adv_results/adv_train --output_json imdb.json --discriminator_checkpoint Bert-Victim/imdb_checkpoint --sim_score_threshold 0.5 
train
python3 train_victim.py --dataset_path imdb --result_path adv_results/imdb_eval.json --num_labels 2 --target_model_path textattack/bert-base-uncased-imdb --checkpoint adv_training/imdb 
eval
python3 attack_classification.py --mode eval --dataset_path data/imdb  --num_labels 2 --target_model_path adv_training/imdb --counter_fitting_embeddings_path data/counter-fitted-vectors.txt --counter_fitting_cos_sim_path data/cos_sim_counter_fitting.npy --output_dir adv_results/adv_train --output_json imdb_eval.json --discriminator_checkpoint Bert-Victim/imdb_checkpoint --sim_score_threshold 0.5


yelp 
generate
python3 attack_classification.py --mode eval --dataset_path yelp_polarity  --num_labels 2 --target_model_path textattack/bert-base-uncased-yelp-polarity --counter_fitting_embeddings_path data/counter-fitted-vectors.txt --counter_fitting_cos_sim_path data/cos_sim_counter_fitting.npy --output_dir adv_results/adv_train --output_json yelp.json --discriminator_checkpoint Bert-Victim/yelp_checkpoint --sim_score_threshold 0.5
train
python3 train_victim.py --dataset_path yelp --result_path adv_results/yelp_eval.json --num_labels 2 --target_model_path textattack/bert-base-uncased-yelp-polarity --checkpoint adv_training/yelp 
eval
python3 attack_classification.py --mode eval --dataset_path data/yelp  --num_labels 2 --target_model_path adv_training/yelp --counter_fitting_embeddings_path data/counter-fitted-vectors.txt --counter_fitting_cos_sim_path data/cos_sim_counter_fitting.npy --output_dir adv_results/adv_train --output_json yelp_eval.json --discriminator_checkpoint Bert-Victim/yelp_checkpoint --sim_score_threshold 0.5



snli
generate
python3 attack_nil.py --mode eval --dataset_path yelp_polarity  --num_labels 2 --target_model_path textattack/bert-base-uncased-yelp-polarity --counter_fitting_embeddings_path data/counter-fitted-vectors.txt --counter_fitting_cos_sim_path data/cos_sim_counter_fitting.npy --output_dir adv_results/adv_train --output_json yelp.json --discriminator_checkpoint Bert-Victim/yelp_checkpoint --sim_score_threshold 0.5
train
python3 train_victim.py --dataset_path snli --result_path adv_results/snli_hypotheses_eval.json --num_labels 3 --target_model_path textattack/bert-base-uncased-snli --checkpoint adv_training/snli  --mode nli
eval
python3 attack_nli.py --mode eval --dataset_path data/snli  --num_labels 3 --target_model_path adv_training/snli --target_component hypotheses --counter_fitting_embeddings_path data/counter-fitted-vectors.txt --counter_fitting_cos_sim_path data/cos_sim_counter_fitting.npy --output_dir adv_results --output_json snli_hypotheses_eval.json --discriminator_checkpoint discriminator_checkpoint_nli/snil_hypothese --sim_score_threshold 0.5 --output_dir adv_results/adv_train


mnli
train
python3 train_victim.py --dataset_path snli --result_path adv_results/mnli_match_hypotheses_eval.json --num_labels 3 --target_model_path textattack/bert-base-uncased-MNLI --checkpoint adv_training/mnli_match  --mode nli
python3 attack_nli.py --mode eval --dataset_path data/mnli_matched  --num_labels 3 --target_model_path adv_training/mnli_match --target_component hypotheses --counter_fitting_embeddings_path data/counter-fitted-vectors.txt --counter_fitting_cos_sim_path data/cos_sim_counter_fitting.npy --output_dir adv_results --output_json mnli_match_hypotheses_eval.json --discriminator_checkpoint discriminator_checkpoint_nli/snil_hypothese --sim_score_threshold 0.5 --output_dir adv_results/adv_train



mnli_mismatch
python3 train_victim.py --dataset_path snli --result_path adv_results/mnli_mismatch_hypotheses_eval.json --num_labels 3 --target_model_path textattack/bert-base-uncased-MNLI --checkpoint adv_training/mnli_mismatch  --mode nli
python3 attack_nli.py --mode eval --dataset_path data/mnli_mismatched  --num_labels 3 --target_model_path adv_training/mnli_mismatch --target_component hypotheses --counter_fitting_embeddings_path data/counter-fitted-vectors.txt --counter_fitting_cos_sim_path data/cos_sim_counter_fitting.npy --output_dir adv_results --output_json mnli_mismatch_hypotheses_eval.json --discriminator_checkpoint discriminator_checkpoint_nli/snil_hypothese --sim_score_threshold 0.5 --output_dir adv_results/adv_train

"""