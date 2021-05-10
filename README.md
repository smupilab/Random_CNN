RandomCNN과 유전 알고리즘을 이용한 네트워크 최적화
=================================================  
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)  
대부분의 신경망을 학습시킬 때 많은 하이퍼파라미터들 설정은 사람이 직접 설정하는 경우가 많다. 하지만 사람이 하이퍼파라미터와 신경망 구조를 어떻게 정해야 신경망의 성능이 좋을지 알기 어렵고 많은 시간을 필요로 한다. 본 연구는 유전 알고리즘을 이용해 신경망 구조 최적화가 사람에 비해 적은 시간을 소비하면서도, 유사하거나 더 좋은 결과를 도출할 수 있음을 보이는 것을 목표로 한다.
### Dataset
* Mnist (Accuracy:99.3%)   
* FashionMnist (Accuracy:93.1%)   
* Cifar10 
* Cifar100  *  
\**는 미구현*

### Running the test
randomCNN network를 `file_writer.py` 이용하여 작성 
`genetic.py`로 유전시켜 최적의 파라미터를 가진 네트워크를 추출  

두 python 파일은 `result` 폴더안에 있습니다.  
linux환경으로 같은 디렉토리에 file_writer.py, genetic.py파일을 넣어 실행시킵니다.  
```
python3 genetic.py
```

### Patch Note
20201124 EarlyStopping 구현  
20201125 최고성능 네트워크 형태를 png파일로 저장
20210410 cifar10 코드 업로드

### Built with
상명대학교 박희민 교수님 연구생  
조한희, 손은영

### Contributors
최지원, 손병욱, 이화경
