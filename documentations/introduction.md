### Problem Definition

기존 Supervised Learing 방식으로 학습된 모델의 파라키터 수가 로 수십~수백억개였던 데에 비하면 매우 규모가 커졌습니다. sefl-supervised 가 내놓는 Pretrained model은 general한 특성을, 즉 다른 fine-tuning만
거치면 다른 모델에 이식하기 용이하다. 

SSL은 CV 보다 NLP 분야에서 많은 성과를 보여왔습니다. 이유는, 이미지에서 uncertainty를 표현하기가 word에 비해서 쉽지 않기 때문입니다. NLP쪽의 Bert는 엄청나게 유명한 모델입니다. 이런 모델들은 SSL 방식으로 unlabeled data에 먼저 학습을 한 후 downstream으로 task의 특성에 adaptation하여
정답이 존재하는 데이터에 fine-tuning 하는 방식이다. SSL 학습은 정답이 없고, 텍스트만 있기 때문에 문장 중간에 단어(토큰)를 masking한 후, 해당 단어를 예측하여 컨텍스트에 맞는 답을 찾아내도록 학습을 시킨다.

자연어 처리 기준으로 볼 때, 각 분야의 고유한 데이터를 바탕으로 학습했을 때보다 높은 정확도를 보인다.

## Human Detection 관련해서 참고하면 좋은 레퍼런스

Histograms of Oriented Gradients for Human Detection, Navneet Dalal, Bill Triggs, International Conference on Computer Vision & Pattern Recognition - June 2005
• http://lear.inrialpes.fr/pubs/2005/DT05/

## Issues in Project 
 SSL 기반 모델들은 매우 큰 사이즈를 가지는데 이런 큰 스케일 모델을 이미지에 적용하여 학습하기가 runtime/memory측면에서 쉽지 않다. 
 contrastive learning 을 활용한 최근의 논문인 SimCLR은 512개의 Cloud TPU v3 cores 써서 4096 batch size 로 모델을 train했다고 한다. -> AWS Telsa T4 인스턴스와 CUAI의 V100 머신과 비교조차 되지 않는 말도 안되는 스펙이다. 
 
 
 참고
 1) https://brunch.co.kr/@advisor/25
 2) https://ai.googleblog.com/2020/04/advancing-self-supervised-and-semi.html
 
