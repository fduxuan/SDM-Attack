

IIMDB

1. train
   python3 rl_classification.py --mode train --dataset_path  imdb  --num_labels 2 --target_model_path textattack/bert-base-uncased-imdb --counter_fitting_embeddings_path data/counter-fitted-vectors.txt --counter_fitting_cos_sim_path data/cos_sim_counter_fitting.npy --output_json imdb_train.json --discriminator_checkpoint imdb_checkpoint_2

   eval

   python3 attack_classification.py --mode eval --dataset_path data/imdb  --num_labels 2 --target_model_path textattack/bert-base-uncased-imdb --counter_fitting_embeddings_path data/counter-fitted-vectors.txt --counter_fitting_cos_sim_path data/cos_sim_counter_fitting.npy --output_json imdb_eval_withsecond.json --discriminator_checkpoint Bert-Victim/imdb_checkpoint --sim_score_threshold 0.5

1000/1000 [55:07<00:00,  3.31s/it, acc=0.027, perturb=0.0648, attack_rate=0.97]
acc: 0.027       perturb: 0.06479286169373857    attack_rate: 0.9703947368421053

checkpoint2

1000/1000 [42:15<00:00,  2.54s/it, acc=0.023, perturb=0.06, attack_rate=0.975]
acc: 0.023       perturb: 0.060047275255877774   attack_rate: 0.9747807017543859

[41:29<00:00,  2.49s/it, acc=0.008, perturb=0.0435, attack_rate=0.991]
acc: 0.008       perturb: 0.04347821279749607    attack_rate: 0.9912280701754386

USE:

python3 USE.py --result_path adv_results/imdb_eval.json

Yelp

train

python3 rl_classification.py --mode train --dataset_path yelp_polarity  --num_labels 2 --target_model_path textattack/bert-base-uncased-yelp-polarity --counter_fitting_embeddings_path data/counter-fitted-vectors.txt --counter_fitting_cos_sim_path data/cos_sim_counter_fitting.npy --output_json yelp_train.json --discriminator_checkpoint Bert-Victim/yelp_checkpoint --sim_score_threshold 0.7

eval

python3 attack_classification.py --mode eval --dataset_path data/yelp  --num_labels 2 --target_model_path textattack/bert-base-uncased-yelp-polarity --counter_fitting_embeddings_path data/counter-fitted-vectors.txt --counter_fitting_cos_sim_path data/cos_sim_counter_fitting.npy --output_json yelp_eval.json --discriminator_checkpoint Bert-Victim/yelp_checkpoint --sim_score_threshold 0.5

[1:06:26<00:00,  3.99s/it, acc=0.018, perturb=0.0771, attack_rate=0.981]

acc: 0.018       perturb: 0.07706129458641162    attack_rate: 0.9814814814814815

acc: 0.024       perturb: 0.07870804957898324    attack_rate: 0.9753086419753086

    USE

    python3 USE.py --result_path adv_results/yelp_eval.json

~/acl_code/Text-Attack/Yelp/test.json

train 特殊的

python3 attack_classification.py --mode train --dataset_path yelp_polarity  --num_labels 2 --target_model_path Victim-TextFooler/yelp --counter_fitting_embeddings_path data/counter-fitted-vectors.txt --counter_fitting_cos_sim_path data/cos_sim_counter_fitting.npy --output_json yelp_train.json --discriminator_checkpoint Bert-Victim/yelp_checkpoint_2 --sim_score_threshold 0.5

python3 attack_classification.py --mode eval --dataset_path data/yelp  --num_labels 2 --target_model_path textattack/bert-base-uncased-yelp-polarity --counter_fitting_embeddings_path data/counter-fitted-vectors.txt --counter_fitting_cos_sim_path data/cos_sim_counter_fitting.npy --output_json yelp_eval.json --discriminator_checkpoint Bert-Victim/yelp_checkpoint_2 --sim_score_threshold 0.5

---

WordCnn

1.mr

train

python3 attack_classification.py --mode train --dataset_path data/mr  --num_labels 2 --target_model_mode CNN --target_model_path WordCnn/mr --word_embeddings_path  WordCnn/glove.6B.200d.txt --counter_fitting_embeddings_path data/counter-fitted-vectors.txt --counter_fitting_cos_sim_path data/cos_sim_counter_fitting.npy --output_dir adv_results/WordCNN --output_json mr_train.json --discriminator_checkpoint discriminator_checkpoint/WordCNN/mr --sim_score_threshold 0.7

eval

python3 attack_classification.py --mode eval --dataset_path data/mr  --num_labels 2 --target_model_mode CNN --target_model_path WordCnn/mr --word_embeddings_path  WordCnn/glove.6B.200d.txt --counter_fitting_embeddings_path data/counter-fitted-vectors.txt --counter_fitting_cos_sim_path data/cos_sim_counter_fitting.npy --output_dir adv_results/WordCNN --output_json mr_eval.json --discriminator_checkpoint discriminator_checkpoint/WordCNN/mr --sim_score_threshold 0.5

imdb

python3 attack_classification.py --mode train --dataset_path imdb  --num_labels 2 --target_model_mode CNN --target_model_path WordCnn/imdb --word_embeddings_path  WordCnn/glove.6B.200d.txt --counter_fitting_embeddings_path data/counter-fitted-vectors.txt --counter_fitting_cos_sim_path data/cos_sim_counter_fitting.npy --output_dir adv_results/WordCNN --output_json imdb_train.json --discriminator_checkpoint discriminator_checkpoint/WordCNN/imdb --sim_score_threshold 0.7

python3 attack_classification.py --mode eval --dataset_path data/imdb  --num_labels 2 --target_model_mode CNN --target_model_path WordCnn/imdb --word_embeddings_path  WordCnn/glove.6B.200d.txt --counter_fitting_embeddings_path data/counter-fitted-vectors.txt --counter_fitting_cos_sim_path data/cos_sim_counter_fitting.npy --output_dir adv_results/WordCNN --output_json imdb_eval.json --discriminator_checkpoint discriminator_checkpoint/WordCNN/imdb --sim_score_threshold 0.5

---

WordLSTM

1.mr

train

python3 attack_classification.py --mode train --dataset_path rotten_tomatoes  --num_labels 2 --target_model_mode LSTM --target_model_path WordLSTM/mr --word_embeddings_path  WordCnn/glove.6B.200d.txt --counter_fitting_embeddings_path data/counter-fitted-vectors.txt --counter_fitting_cos_sim_path data/cos_sim_counter_fitting.npy --output_dir adv_results/WordLSTM --output_json mr_train.json --discriminator_checkpoint discriminator_checkpoint/WordLSTM/mr --sim_score_threshold 0.7

eval

python3 attack_classification.py --mode eval --dataset_path data/mr  --num_labels 2 --target_model_mode LSTM --target_model_path WordLSTM/mr --word_embeddings_path  WordCnn/glove.6B.200d.txt --counter_fitting_embeddings_path data/counter-fitted-vectors.txt --counter_fitting_cos_sim_path data/cos_sim_counter_fitting.npy --output_dir adv_results/WordLSTM --output_json mr_eval.json --discriminator_checkpoint discriminator_checkpoint/WordLSTM/mr --sim_score_threshold 0.5

2. imdb train

python3 attack_classification.py --mode train --dataset_path rotten_tomatoes  --num_labels 2 --target_model_mode LSTM --target_model_path WordLSTM/imdb --word_embeddings_path  WordCnn/glove.6B.200d.txt --counter_fitting_embeddings_path data/counter-fitted-vectors.txt --counter_fitting_cos_sim_path data/cos_sim_counter_fitting.npy --output_dir adv_results/WordLSTM --output_json imdb_train.json --discriminator_checkpoint discriminator_checkpoint/WordLSTM/imdb --sim_score_threshold 0.7

python3 attack_classification.py --mode eval --dataset_path data/imdb  --num_labels 2 --target_model_mode LSTM --target_model_path WordLSTM/imdb --word_embeddings_path  WordCnn/glove.6B.200d.txt --counter_fitting_embeddings_path data/counter-fitted-vectors.txt --counter_fitting_cos_sim_path data/cos_sim_counter_fitting.npy --output_dir adv_results/WordLSTM --output_json imdb_eval.json --discriminator_checkpoint discriminator_checkpoint/WordLSTM/imdb --sim_score_threshold 0.5

---

SNLI

python3 attack_nli.py --mode train --dataset_path data/snli  --num_labels 3 --target_model_path textattack/bert-base-uncased-snli --target_component hypotheses --counter_fitting_embeddings_path data/counter-fitted-vectors.txt --counter_fitting_cos_sim_path data/cos_sim_counter_fitting.npy --output_dir adv_results --output_json snli_hypotheses.json --discriminator_checkpoint discriminator_checkpoint_nli/snil_hypothese --sim_score_threshold 0.5

python3 attack_nli.py --mode eval --dataset_path data/snli  --num_labels 3 --target_model_path textattack/bert-base-uncased-snli --target_component hypotheses --counter_fitting_embeddings_path data/counter-fitted-vectors.txt --counter_fitting_cos_sim_path data/cos_sim_counter_fitting.npy --output_dir adv_results --output_json snli_hypotheses_eval.json --discriminator_checkpoint discriminator_checkpoint_nli/snil_hypothese --sim_score_threshold 0.5   --perturb_ratio 1

snli premises

python3 attack_nli.py --mode eval --dataset_path data/snli  --num_labels 3 --target_model_path textattack/bert-base-uncased-snli --target_component premises --counter_fitting_embeddings_path data/counter-fitted-vectors.txt --counter_fitting_cos_sim_path data/cos_sim_counter_fitting.npy --output_dir adv_results --output_json snli_premises_eval.json --discriminator_checkpoint discriminator_checkpoint_nli/snli_premises --sim_score_threshold 0.5 --perturb_ratio 0.5

mnli-matched  hypothese

python3 attack_nli.py --mode train --dataset_path data/mnli_matched  --num_labels 3 --target_model_path textattack/bert-base-uncased-MNLI --target_component hypotheses --counter_fitting_embeddings_path data/counter-fitted-vectors.txt --counter_fitting_cos_sim_path data/cos_sim_counter_fitting.npy --output_dir adv_results --output_json mnli_match_hypotheses_train.json --discriminator_checkpoint discriminator_checkpoint_nli/mnli_hypothese

python3 attack_nli.py --mode eval --dataset_path data/mnli_matched  --num_labels 3 --target_model_path textattack/bert-base-uncased-MNLI --target_component hypotheses --counter_fitting_embeddings_path data/counter-fitted-vectors.txt --counter_fitting_cos_sim_path data/cos_sim_counter_fitting.npy --output_dir adv_results --output_json mnli_match_hypotheses_eval.json --discriminator_checkpoint discriminator_checkpoint_nli/mnli_hypothese --sim_score_threshold 0.5

mnli-mismatched  hypothese

python3 attack_nli.py --mode train --dataset_path data/mnli_mismatched  --num_labels 3 --target_model_path textattack/bert-base-uncased-MNLI --target_component hypotheses --counter_fitting_embeddings_path data/counter-fitted-vectors.txt --counter_fitting_cos_sim_path data/cos_sim_counter_fitting.npy --output_dir adv_results --output_json mnli_mismatch_hypotheses_eval.json --discriminator_checkpoint discriminator_checkpoint_nli/snil_hypothese --sim_score_threshold 0.5 --perturb_ratio 1

python3 attack_nli.py --mode eval --dataset_path data/mnli_mismatched  --num_labels 3 --target_model_path textattack/bert-base-uncased-MNLI --target_component hypotheses --counter_fitting_embeddings_path data/counter-fitted-vectors.txt --counter_fitting_cos_sim_path data/cos_sim_counter_fitting.npy --output_dir adv_results --output_json mnli_mismatch_hypotheses_eval.json --discriminator_checkpoint discriminator_checkpoint_nli/snil_hypothese --sim_score_threshold 0.5 --perturb_ratio 1

---

SNLI hypotheses infersent

python3 attack_nli.py --mode eval --dataset_path data/snli  --num_labels 3 --target_model_mode InferSent  --target_model_path Victim-TextFooler/SNLI/pytorch_model.bin --target_component hypotheses --counter_fitting_embeddings_path data/counter-fitted-vectors.txt --counter_fitting_cos_sim_path data/cos_sim_counter_fitting.npy --word_embeddings_path  Victim-TextFooler/glove.840B.300d.txt --output_dir adv_results/InferSent --output_json snli_hypotheses_eval.json --discriminator_checkpoint discriminator_checkpoint_nli/snil_hypothese --sim_score_threshold 0.5 --perturb_ratio 1

---