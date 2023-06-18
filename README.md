# Topic-modeling

![image](https://github.com/ugiugi0823/topic-modeling/assets/106899647/9f515358-a9c3-48cb-93d4-953145f7c6e7)


### How to Run with Colab


0. 드라이브 연결은 필수입니다.
```
if google_drive:
  drive.mount('/content/drive')
```



1.
```
!git clone https://github.com/ugiugi0823/topic-modeling.git
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















