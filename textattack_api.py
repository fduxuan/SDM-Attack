# -*- coding: utf-8 -*-
'''
Created on: 2022-12-31 04:16:42
LastEditTime: 2023-01-14 15:31:09
Author: fduxuan

Desc:  

'''
import textattack
from textattack.attack_recipes import BERTAttackLi2020, TextFoolerJin2019, CLARE2020, BAEGarg2019, A2TYoo2021
import transformers
import collections
import argparse
import os
from util import *
import tqdm
import json
import dataloader
from model import *
from datetime import datetime
# from USE import USE
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True, 
                        help="Which dataset to attack.")
    parser.add_argument("--num_labels", type=int, default=3,
                        help="How many classes for classification.")
    parser.add_argument("--target_model_path", type=str, required=True,
                        help="Victim Models checkpoint")
    parser.add_argument("--output_dir", type=str, default='adv_results',
                        help="The output directory where the attack results will be written.")
    parser.add_argument("--output_json", type=str, default='result.json',
                        help="The attack results.")
    parser.add_argument("--method", type=str, default='bert-attack', help='bert-attack/textfooler/clare')
    parser.add_argument("--mode", type=str, default='nli', help='nli/classification')
    
    args = parser.parse_args()
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        print("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    else:
        os.makedirs(args.output_dir, exist_ok=True)
    
    info('Load Victim!')
    victim = transformers.AutoModelForSequenceClassification.from_pretrained(args.target_model_path, num_labels=args.num_labels).to('cuda')
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.target_model_path)
    model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(victim, tokenizer)
    if args.method == 'bert-attack':
        attack = BERTAttackLi2020.build(model_wrapper)
    elif args.method == 'clare':
        attack = CLARE2020.build(model_wrapper)
    elif args.method == 'bae':
        attack  = BAEGarg2019.build(model_wrapper)
    elif args.method == 'a2t':
        attack = A2TYoo2021.build(model_wrapper)
    else:
        attack = TextFoolerJin2019.build(model_wrapper)
        
    info('load data!')
    
    res = []
    if args.mode == 'nli':  
        data = read_data(args.dataset_path)

        info('Start attacking!')
        for d in tqdm.tqdm(data):
            t1 = datetime.now()
            input_text = collections.OrderedDict(premise=" ".join(d.premises),
                                                hypothesis=" ".join(d.hypotheses))
            label = d.label
            attack_result = attack.attack(input_text, label)
            # print(attack_result.original_text("ansi"))
            # print(attack_result.perturbed_text("ansi"))
            result = str(attack_result).split('\n')
            # print(attack_result)
            ans = {}
            change = []
            # print(result)
            if len(result) > 6: # 成功
                
                ans['org_seq'] = result[3].split('Hypothesis:')[1].strip()
                ans['adv_seq'] = result[6].split('Hypothesis:')[1].strip()
                ans['result'] = result[0]
                a = ans['org_seq'].split()
                b = ans['adv_seq'].split()
                for j in range(len(a)):
                    if a[j] != b[j]:
                        change.append([j, f"{a[j]}-->{b[j]}"])
                ans['change'] = change
                t2 = datetime.now()
                ans['second'] = (t2-t1).seconds
                res.append(ans)
                
    else:
        
        texts, labels = dataloader.read_corpus(args.dataset_path)
        data = [ClassificationItem(words=texts[i], label=labels[i]) for i in range(len(labels))]
        info('Start attacking!')
        for d in tqdm.tqdm(data):
            t1 = datetime.now()
            input_text = " ".join(d.words)
            label = d.label
            attack_result = attack.attack(input_text, label)
          
            result = str(attack_result).split('\n')
            # print(attack_result)
            ans = {}
            # print(result)
            # print(result)
            if len(result) > 4: # 成功
                ans['org_seq'] = result[2]
                ans['adv_seq'] = result[4]
                ans['result'] = result[0]
                t2 = datetime.now()
                ans['second'] = (t2-t1).seconds
                res.append(ans)
    save_path = args.output_dir+"/"+args.output_json
    
    json.dump(res, open(save_path, 'w'))
   
def attack():
    model = "textattack/bert-base-uncased-rotten_tomatoes"
    victim = transformers.AutoModelForSequenceClassification.from_pretrained(model, num_labels=2).to('cuda')
    tokenizer = transformers.AutoTokenizer.from_pretrained(model)
    model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(victim, tokenizer)
    text = "davis is so enamored of her own creation that she ca n't see how insufferable the character is"
    attack = A2TYoo2021.build(model_wrapper)
    # attack = TextFoolerJin2019.build(model_wrapper)
    # data = ClassificationItem(words=text.split(), label=0)
    attack_result = attack.attack(text, 0)
    print(attack_result)
    
def test():
    
    model = "textattack/bert-base-uncased-rotten_tomatoes"

    text = "davis is so enamored of her own creation that she ca n't see how insufferable the character is"
    # textfooler
    # text2 = "davis is well enamored of her own infancy that she could n't admire how infernal the idiosyncrasies is"
    # bertattack
    # text2 = "davis is often enamoted of her own generation that she ca n't see how insuffoure the queen is"
    # a2t
    text2 = "davis is so enamored of her own institution that she ca n't behold how unforgivable the hallmark is"
    # rl-attack
    # text2 = "davis is so captivated of her own creation that she ca n't see how indefensible the character is"
    data = ClassificationItem(words=text.split(), label=0)
    victim = VictimModelForTransformer(model, 2)
    print(victim.text_pred([text.split(), text2.split()]))
    
    # print(attack_result)
    from USE import USE
    use = USE("dd", "data/USE_Model", "dd")
    score = use.semantic_sim([text], [text2])
    perturb = 0
    word1 = text.split()
    word2 = text2.split()
    for i in range(len(word1)):
        if word1[i] != word2[i]:
            perturb += 1
    
    print(score, perturb/len(word1))


if __name__ == "__main__":
    # test()
    test()
    # attack()

"""
snli
1. bert-attack
python3 textattack_api.py --dataset_path data/snli --target_model_path textattack/bert-base-uncased-snli --output_dir textattack_result/bertattack --output_json snli_pre2.json

2. textfooler
python3 textattack_api.py --dataset_path data/snli --target_model_path textattack/bert-base-uncased-snli --output_dir textattack_result/textfooler --output_json snli.json --method textfooler

3. snli
python3 textattack_api.py --dataset_path data/snli --target_model_path textattack/bert-base-uncased-snli --output_dir textattack_result/bae --output_json snli.json --method bae

5. a2t
python3 textattack_api.py --dataset_path data/snli --target_model_path textattack/bert-base-uncased-snli --output_dir textattack_result/a2t --output_json snli.json --method a2t


mnli matched
1. bert-attack
python3 textattack_api.py --dataset_path data/mnli_matched --target_model_path textattack/bert-base-uncased-MNLI --output_dir textattack_result/bertattack --output_json mnli_matched_pre.json

2. textfooler
python3 textattack_api.py --dataset_path data/mnli_matched --target_model_path textattack/bert-base-uncased-MNLI --output_dir textattack_result/textfooler --output_json mnli_matched_pre.json --method textfooler

3.bae
python3 textattack_api.py --dataset_path data/mnli_matched --target_model_path textattack/bert-base-uncased-MNLI --output_dir textattack_result/bae --output_json mnli_matched.json --method bae

5. a2t
python3 textattack_api.py --dataset_path data/mnli_matched --target_model_path textattack/bert-base-uncased-MNLI --output_dir textattack_result/a2t --output_json mnli_matched.json --method a2t


mnli mismatched
1. bert-attack
python3 textattack_api.py --dataset_path data/mnli_mismatched --target_model_path textattack/bert-base-uncased-MNLI --output_dir textattack_result/bertattack --output_json mnli_mismatched.json

2. textfooler
python3 textattack_api.py --dataset_path data/mnli_mismatched --target_model_path textattack/bert-base-uncased-MNLI --output_dir textattack_result/textfooler --output_json mnli_mismatched.json --method textfooler

3. bae
python3 textattack_api.py --dataset_path data/mnli_mismatched --target_model_path textattack/bert-base-uncased-MNLI --output_dir textattack_result/bae --output_json mnli_mismatched.json --method bae

5. a2t
python3 textattack_api.py --dataset_path data/mnli_mismatched --target_model_path textattack/bert-base-uncased-MNLI --output_dir textattack_result/a2t --output_json mnli_mismatched.json --method a2t


yelp
1. bert-attack
python3 textattack_api.py --dataset_path data/yelp --target_model_path textattack/bert-base-uncased-yelp-polarity --num_labels 2 --output_dir textattack_result/bertattack --output_json yelp2.json --mode classification

2. textfooler
python3 textattack_api.py --dataset_path data/yelp --target_model_path textattack/bert-base-uncased-yelp-polarity --num_labels 2 --output_dir textattack_result/textfooler --output_json yelp.json --mode classification --method textfooler

3. clare
python3 textattack_api.py --dataset_path data/yelp --target_model_path textattack/bert-base-uncased-yelp-polarity --num_labels 2 --output_dir textattack_result/clare --output_json yelp.json --mode classification --method clare

4. bae
python3 textattack_api.py --dataset_path data/yelp --target_model_path textattack/bert-base-uncased-yelp-polarity --num_labels 2 --output_dir textattack_result/bae --output_json yelp.json --mode classification --method bae

5. a2t
python3 textattack_api.py --dataset_path data/yelp --target_model_path textattack/bert-base-uncased-yelp-polarity --num_labels 2 --output_dir textattack_result/a2t --output_json yelp.json --mode classification --method a2t


imdb
1. bert-attack
python3 textattack_api.py --dataset_path data/imdb --target_model_path textattack/bert-base-uncased-imdb --num_labels 2 --output_dir textattack_result/bertattack --output_json imdb2.json --mode classification

2.textfooler
python3 textattack_api.py --dataset_path data/imdb --target_model_path textattack/bert-base-uncased-imdb --num_labels 2 --output_dir textattack_result/textfooler --output_json imdb.json --mode classification --method textfooler

3. clare
python3 textattack_api.py --dataset_path data/imdb --target_model_path textattack/bert-base-uncased-imdb --num_labels 2 --output_dir textattack_result/clare --output_json imdb.json --mode classification --method clare

4.bae
python3 textattack_api.py --dataset_path data/imdb --target_model_path textattack/bert-base-uncased-imdb --num_labels 2 --output_dir textattack_result/bae --output_json imdb.json --mode classification --method bae

5. a2t
python3 textattack_api.py --dataset_path data/imdb --target_model_path textattack/bert-base-uncased-imdb --num_labels 2 --output_dir textattack_result/a2t --output_json imdb.json --mode classification --method a2t

ag
1. bert-attack
python3 textattack_api.py --dataset_path data/ag --target_model_path textattack/bert-base-uncased-ag-news --num_labels 4 --output_dir textattack_result/bertattack --output_json ag.json --mode classification
3. bae
python3 textattack_api.py --dataset_path data/ag --target_model_path textattack/bert-base-uncased-ag-news --num_labels 4 --output_dir textattack_result/bae --output_json ag.json --mode classification --method bae

5. a2t
python3 textattack_api.py --dataset_path data/ag --target_model_path textattack/bert-base-uncased-ag-news --num_labels 4 --output_dir textattack_result/a2t --output_json ag.json --mode classification --method a2t


mr 
1. bert-attack
python3 textattack_api.py --dataset_path data/mr --target_model_path textattack/bert-base-uncased-rotten_tomatoes --num_labels 2 --output_dir textattack_result/bertattack --output_json mr.json --mode classification

2. textfooler
python3 textattack_api.py --dataset_path data/mr --target_model_path textattack/bert-base-uncased-rotten_tomatoes --num_labels 2 --output_dir textattack_result/textfooler --output_json mr.json --mode classification --method textfooler

3. bae
python3 textattack_api.py --dataset_path data/mr --target_model_path textattack/bert-base-uncased-rotten_tomatoes --num_labels 2 --output_dir textattack_result/bae --output_json mr.json --mode classification --method bae

5. a2t
python3 textattack_api.py --dataset_path data/mr --target_model_path textattack/bert-base-uncased-rotten_tomatoes --num_labels 2 --output_dir textattack_result/a2t --output_json mr.json --mode classification --method a2t

"""