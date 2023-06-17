import os
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from utils import get_db

def get_topic_model(args):
  output_path = args.output_path
  model_name = args.model_name.replace("/", "-")
  
  model_s = SentenceTransformer(f'{output_path}/{model_name}')
  topic_model = BERTopic(embedding_model=model_s)



  docs = get_db(args)
  docc = 'raw_all'

  topics, probabilities = topic_model.fit_transform(docs)
  topic_model.save(f"{output_path}/{docc}_model")
  return print('get_topic_model 종료')
