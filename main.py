import argparse
from train import Get_train
from utils import setup

if __name__ == '__main__':
  p = argparse.ArgumentParser()
  p.add_argument("--output_path", type=str, default="/content/drive/MyDrive/LDA/Robert_sentence_transformers/out/training_klue_sts_", help="모델 파일 저장되는 경로를 넣어주세요!") 
  p.add_argument("--model_name", type=str, default="ugiugi/inisw08-DistilBERT-mlm-adamw_torch", help="원하는 임베이딩 모델 이름을 넣어주세요(허깅페이스)")  
  p.add_argument("--max_seq_length", type=int, default=512, help="max_seg_length 를 정해주세요")  
  p.add_argument("--batch_size", type=int, default=32,  help="batch_size") 
  p.add_argument("--n_epoch", type=int, default=4, help="epoch 수") 
  p.add_argument("--use_amp", type=bool, default=True, help="pytorch 1.6 이상이면, AMP 를 사용") 
  p.add_argument("--cls_token",type=bool, default=True, help="cls_token 을 쓸거면 true 해주세요!") 
  p.add_argument("--do_lower", type=bool, default=True, help="소문자 적용") 


  
  args = p.parse_args()
  setup(args)
  Get_train(args)




