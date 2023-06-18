# run.sh
pip install transformers
pip install wandb
gdown '1wVh8tP0XcOuabv9EUa6hnXiykqjTqQ1t&confirm=t'
unzip database_csv.zip -d /content/tweet-sa-robert/data

# 💙 3가지를 수정해 주시면 정상적으로 돌아가요! 💙


# 1. 허깅페이스 로그인 토큰
# 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
huggingface-cli login --token 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



# 2. wandb 로그인 토큰
# inisw (중요)
# 2be184e31a96c722bfebdfe35f726042eb8e526c
# 현욱
# 122f007f67ba33fd04a03ee9b81489dfb42264a6
wandb login --relogin '122f007f67ba33fd04a03ee9b81489dfb42264a6'




python main.py \
  --output_path "/content/drive/MyDrive/inisw08/bertopic" \
  --model_name "ugiugi/inisw08-T5-mlm-adafactor_test" \
  --db_name "preproc_6_15.db" \
  --repo_name "T5-BERTopic-bs32-ep100" \
  --max_seq_length 512 \
  --batch_size 32 \
  --n_epoch 100 \
  --use_amp \
  --cls_token \
  --do_lower \
  --drive 
  
    
