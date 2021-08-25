# BYOL : Self-supervised Representation learning method
## Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning (BYOL)
 ### by Google Deepmind
 
 BYOL을 이해하기 위해서는 Representation learning과 Self-supervised learning에 대해 알고 있어야 하는데, Representation learning은 입력 데이터의 특징을 잘 표현할 수 있도록 학습하는 것이고, Self-supervised learning은 레이블이 없는 데이터를 활용하여 모델이 좋은 특징을 추출할 수 있도록 학습하는 방법론. 
 
### BYOL이 나오게 된 계기
 출처: https://2-chae.github.io/category/2.papers/26 

"" 많은 성공적인 SSL 기법들은 제프리 힌턴 교수의 논문 
Self-organizing neural network that discovers surfaces in random-dot stereograms에서 나온 *cross-view prediction framework*를 베이스로 만들어졌다고 한다.""
기존 SSL 방식들이 Negative sample에 의존했던 이유는 훈련 도중 발생하는 collapse 방지 위해서였다. BYOL은 MoCo에서의 momentum encoder 업데이트와 마찬가지로, negative sample들의 representation을 consistent하게 보존하기 위해서 두 가지 네트워크가 시간차를 두어 parameter update를 수행하는 이색적인 방식을 이용한다고 볼 수 있다.

### BYOL의 특이점과 어떤 요소가 Representation Collapse를 방지하는가
1.  Loss function 
  prediciton 부분이 asymmetric
  ![image](https://user-images.githubusercontent.com/75107070/130728545-9be5edf2-48fa-42aa-86d9-98748b2d08b9.png)

2.  간단하면서도 특이한 모델 구조 
   네트워크의 아주 작은 부분을 데이터에 대해 학습시켰고 학습된 네트워크로 뽑아낸 임의의 representation을 target으로 삼아, 또 다른 random initialize된 네트워크는 더 좋은 representation을 배웠다.
   ![image](https://user-images.githubusercontent.com/75107070/130730634-61434b28-bf3d-428d-9b7a-b9932dc6eb72.png)


![image](https://user-images.githubusercontent.com/75107070/130727687-b8224fe6-932e-480d-99f1-4cac9be4e36a.png)
 

BYOL을 사용하는 목적은 downstream task에서 사용될 representation y를 학습하는 것이다. ResNet-50 또는 ResNet-200

### Usage
![image](https://user-images.githubusercontent.com/75107070/130728107-70463e21-fc64-4e62-b95b-84d0d5c3b236.png)
*figure from https://paperswithcode.com/method/byol*
의문) ve-id 문제 해결 시에는 객체 감지는 안해도 되나? veri-wild 데이터셋 보니까 차량이 예쁘게 중앙에 배치 안되어 있는 경우도 존재. 
BYOL을 자동차 관련해서, object detection 관련으로도 확장되어야 하는데, MoCo와 같은 방식으로 Faster-RCNN사용하는 setup으로 VOC2012에대해 검증되었음 
ToDo
1. BYOL code implementation
2. pretrain and validation on transfer-learned model for STL-10
  
  우선 ImageNet 데이터셋에 대해선, ImageNet unlabeled 데이터셋으로 Self Supervised Learning 방식으로 Pre-training을 시킨 뒤, feature extractor(encoder)를 freeze 시키고, linear classifier를 학습시키는 linear evaluation 실험과, 일부분의 labeled training set을 가지고 feature extractor를 fine-tuning 시키는(do not freeze) semi-supervised training(aka. 
semi-supervised and transfer benmark) 실험을 수행하였습니다. 마찬가지로 두가지 방식의 experiment settings 존재.


3. Veri-WILD 차종 데이터 전처리 조사 
   1) *The Devil is in Details: Self-Supervised Attention for Vehicle Re-identification
   2) *Self-paced Contrastive Learning with Hybrid Memory for Domain Adaptive Object Re-ID
4. pretrain and validation on transfer-learned model for 차종 데이터
