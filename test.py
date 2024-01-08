
import argparse
import math
import re

import torch
from tqdm import tqdm
import numpy as np
# import language_check

from bert_score import score
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from USE import USE


def process_string(string):
    string = re.sub("( )(\'[(m)(d)(t)(ll)(re)(ve)(s)])", r"\2", string)
    string = re.sub("(\d+)( )([,\.])( )(\d+)", r"\1\3\5", string)
    # U . S . -> U.S.
    string = re.sub("(\w)( )(\.)( )(\w)( )(\.)", r"\1\3\5\7", string)
    # reduce left space
    string = re.sub("( )([,\.!?:;)])", r"\2", string)
    # reduce right space
    string = re.sub("([(])( )", r"\1", string)
    string = re.sub("s '", "s'", string)
    # reduce both space
    string = re.sub("(')( )(\S+)( )(')", r"\1\3\5", string)
    string = re.sub("(\")( )(\S+)( )(\")", r"\1\3\5", string)
    string = re.sub("(\w+) (-+) (\w+)", r"\1\2\3", string)
    string = re.sub("(\w+) (/+) (\w+)", r"\1\2\3", string)
    # string = re.sub(" ' ", "'", string)
    return string

def calculate_ppl(texts, model, tokenizer):
    model.eval()
    eval_loss = 0
    nb_eval_steps = 0

    with torch.no_grad():
        for text in tqdm(texts):
            text = process_string(text)
            input_ids = torch.tensor(tokenizer.encode(text, add_special_tokens=True))
            if len(input_ids) < 2:
                continue
            input_ids = input_ids.cuda()
            outputs = model(input_ids, labels=input_ids)
            lm_loss = outputs[0]
            eval_loss += lm_loss.mean().item()
            nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))
    return perplexity.item()

def evaluate(refs, hypos):
    # GPT2 model for perpelexity calculation
    # tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    # model = GPT2LMHeadModel.from_pretrained('gpt2').cuda()

    # print("Evaluating references ppl ...")
    # ref_ppl = calculate_ppl(refs, model, tokenizer)
    # print("Reference ppl: %.4f" % ref_ppl)

    # print("Evaluating hypothesis ppl ...")
    # hy_ppl = calculate_ppl(hypos, model, tokenizer)
    # print("Hypothesis ppl: %.4f" % hy_ppl)

    # # BERT SCORE:
    # P, R, F1 = score(hypos, refs, lang='en', verbose=True)
    # print('BERT score: ', F1.mean())

    # USE similarity score
    use = USE('dd', 'data/USE_Model', 'dd')
    print('Evaluating USE similarity:')
    sim_score = []
    perturb_rate = []
    success = 0
    for i in tqdm(range(len(refs))):
        
        
        org_words = refs[i].split()
        adv_words = hypos[i].split()
        change = 0
        for j in range(len(org_words)):
            if org_words[j] != adv_words[j]:
                change += 1
        perturb = change / len(org_words)
        if perturb > 0.25: # 25---nli 0.4 classi
            continue
        success += 1
        sim_score.append(use.semantic_sim([refs[i]], [hypos[i]])[0][0])
        perturb_rate.append(perturb)
    sim_score = np.mean(sim_score)
    perturb = np.mean(perturb_rate)
    print('USE sim score: ', sim_score)
    print('perturb_Rate: ', perturb)
    print('A-rate:', success/1000)
    # evalute number of grammar errors
    
    # tool = language_check.LanguageTool('en-US')
    # grammar_diffs = []
    # for i in tqdm(range(len(refs))):
    #     grammar_ref = len(tool.check(process_string(refs[i])))
    #     grammar_hypo = len(tool.check(process_string(hypos[i])))
    #     grammar_diffs.append(grammar_hypo - grammar_ref)
    # gramar_err = np.mean(grammar_diffs)
    # print("number of grammar difference: ", gramar_err)
    # print('\n')
    
    # return ref_ppl, hy_ppl, F1.mean(), sim_score, gramar_err


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--eval_file",
                            type=str,
                            required=True,
                            help="adversarial output file for evaluation.")
    argparser.add_argument("--mode",
                            type=str,
                            default='classification',
                            help="Path to the USE encoder cache.")
    args = argparser.parse_args()
    # use = USE(args.USE_cache_path)

    refs = []
    hypos = []

    print("Evaluating %s", args.eval_file)
    if args.mode == 'classification':
        with open(args.eval_file, 'r', encoding='utf8') as f:
            for line in f:
                if "orig sent" in line:
                    refs.append(line.split('\t')[1].strip())
                if "adv sent" in line:
                    hypos.append(line.split('\t')[1].strip())
        assert len(refs) == len(hypos)
        evaluate(refs, hypos)
    else:
        with open(args.eval_file, 'r', encoding='utf8') as f:
            for line in f:
                if "orig hypothesis" in line:
                    refs.append(line.split('\t')[1].strip())
                if "adv hypothesis" in line:
                    hypos.append(line.split('\t')[1].strip())
        evaluate(refs, hypos)