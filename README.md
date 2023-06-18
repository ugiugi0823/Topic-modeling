# Topic-modeling



## How to Run with Colab


[1]
```
!git clone https://github.com/ugiugi0823/topic-modeling.git
```




```


```





```
  p = argparse.ArgumentParser()
  p.add_argument("--output_path", type=str, default="/content/drive/MyDrive/inisw08/bertopic", help="모델 파일 저장되는 경로를 넣어주세요!") 
  p.add_argument("--model_name", type=str, default="ugiugi/inisw08-T5-mlm-adafactor_test", help="원하는 임베이딩 모델 이름을 넣어주세요(허깅페이스)")  
  p.add_argument("--max_seq_length", type=int, default=512, help="max_seg_length 를 정해주세요")  
  p.add_argument("--batch_size", type=int, default=32,  help="batch_size") 
  p.add_argument("--n_epoch", type=int, default=100, help="epoch 수") 
  p.add_argument("--use_amp", type=bool, default=True, help="pytorch 1.6 이상이면, AMP 를 사용") 
  p.add_argument("--cls_token",type=bool, default=True, help="cls_token 을 쓸거면 true 해주세요!") 
  p.add_argument("--do_lower", type=bool, default=True, help="소문자 적용") 
  p.add_argument("--db_name", type=str, default="preproc_6_15.db", help="db 이름") 
  p.add_argument("--drive", type=bool, default=False, help="Drive 저장하고 싶으면 True")
  p.add_argument("--repo_name", type=str, default="T5-BERTopic-bs32-ep100", help="db 이름")  
  


```


```
import gc
gc.collect()
!python main.py \
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


```
