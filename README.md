# Topic-modeling

![image](https://github.com/ugiugi0823/topic-modeling/assets/106899647/9f515358-a9c3-48cb-93d4-953145f7c6e7)


### 코랩으로 쉽게 돌려보기
- 순서대로 돌려주세요!


0. 드라이브 연결은 필수입니다.
```
from google.colab import drive
drive.mount('/content/drive')
```



1.
```
!git clone https://github.com/ugiugi0823/topic-modeling.git
%cd topic-modeling
```



2. 시간이 오래 걸립니다.!
```
%cd /content/topic-modeling
!bash run.sh
```


3. 결과 확인
```
from PIL import Image
import matplotlib.pyplot as plt
image = Image.open('/content/drive/MyDrive/inisw08/bertopic/barchart/ugiugi-inisw08-T5-mlm-adafactor_test_barchart.png')
plt.imshow(image)
plt.axis('off')  
plt.show()
```















