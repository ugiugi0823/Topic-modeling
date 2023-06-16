import argparse
from train import Get_train
from utils import setup
from get_topic_model import Get_topic_model
from get_topic import Get_topic

if __name__ == '__main__':
  p = argparse.ArgumentParser()
  p.add_argument("--output_path", type=str, default="/content/drive/MyDrive/wxxk/bertopic", help="모델 파일 저장되는 경로를 넣어주세요!") 
  p.add_argument("--model_name", type=str, default="ugiugi/inisw08-T5-mlm-adafactor_test", help="원하는 임베이딩 모델 이름을 넣어주세요(허깅페이스)")  
  p.add_argument("--max_seq_length", type=int, default=512, help="max_seg_length 를 정해주세요")  
  p.add_argument("--batch_size", type=int, default=32,  help="batch_size") 
  p.add_argument("--n_epoch", type=int, default=4, help="epoch 수") 
  p.add_argument("--use_amp", type=bool, default=True, help="pytorch 1.6 이상이면, AMP 를 사용") 
  p.add_argument("--cls_token",type=bool, default=True, help="cls_token 을 쓸거면 true 해주세요!") 
  p.add_argument("--do_lower", type=bool, default=True, help="소문자 적용") 
  p.add_argument("--db_name", type=str, default="preproc_6_15.db", help="db 이름") 
  



  
  args = p.parse_args()
  setup(args)
  Get_train(args)
  Get_topic_model(args)
  Get_topic(args)




