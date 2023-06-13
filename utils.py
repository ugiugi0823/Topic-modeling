# GLUE STS 내 훈련, 검증 데이터 예제 변환
from sentence_transformers.readers import InputExample

def Preprocess(datasets):
  train_samples = []
  dev_samples = []
  test_samples = []
  for phase in ["train", "validation"]:
      examples = datasets[phase]
      

      for example in examples:
          score = float(example["label"]) / 5.0  # 0.0 ~ 1.0 스케일로 유사도 정규화

          inp_example = InputExample(
              texts=[example["sentence1"], example["sentence2"]],
              label=score,
          )

          if phase == "validation":
              dev_samples.append(inp_example)
          else:
              train_samples.append(inp_example)

  return train_samples, dev_samples
