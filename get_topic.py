from bertopic import BERTopic
import os

def Get_topic(args):

  model_name = args.model_name.replace("/", "-")
  output_path = args.output_path
  load_model = BERTopic.load(f"{output_path}/raw_all_model")
  barchart = load_model.visualize_barchart()

  return barchart.write_image(f"{output_path}/{model_name}_barchart.png")
