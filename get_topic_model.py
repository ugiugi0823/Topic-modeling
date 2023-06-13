import os
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from utils import get_db

def Get_topic_model(args):
  output_path = args.output_path
  output_path = output_path.split('/')[-1]

  folder_path = f"./{output_path}"  # 'out' 폴더의 경로
  file_list = os.listdir(folder_path)



  model_s = SentenceTransformer(f'./{output_path}/{file_list[1]}')
  topic_model = BERTopic(embedding_model=model_s)



  docs = get_db()
  docc = 'raw_all'

  topics, probabilities = topic_model.fit_transform(docs)
  topic_model.save(f"./{output_path}/{docc}_model")
  return print('get_topic_model 종료')
