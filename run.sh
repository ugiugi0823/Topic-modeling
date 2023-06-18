pip install bertopic
pip install sentence-transformers datasets
pip install -U kaleido
gdown '1wVh8tP0XcOuabv9EUa6hnXiykqjTqQ1t&confirm=t'
unzip database_csv.zip -d /content/topic-modeling/data



python main.py \
  --output_path "/content/drive/MyDrive/inisw08/bertopic" \
  --model_name "ugiugi/inisw08-RoBERT-mlm-adamw_torch_bs16" \
  --max_seq_length 512 \
  --batch_size 64 \
  --n_epoch 1 \
  --use_amp True \
  --cls_token True \
  --do_lower True \
  --db_name "preproc_6_2.csv" \
  --drive True
