from bertopic import BERTopic
import os

def Get_topic(args):
  output_path = 'out'
  folder_path = f"./{output_path}"  # 'out' 폴더의 경로
  file_list = os.listdir(folder_path)


  load_model = BERTopic.load(f"./{output_path}/raw_all_model")
  barchart = load_model.visualize_barchart()

  return barchart.write_image(f"./barchart.png")
