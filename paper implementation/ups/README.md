pseudo-labeling 기반 방법론은 high confidence를 가지는 unlabeled samples을 선별해 training targets (pseudo-labels)으로 사용합니다. Decision boundary에서의 data point 밀도를 줄이므로써 entropy를 최소화하는 것으로 볼 수 있습니다.

## Pseudo-labeling (PL) vs. Consistency-regularization (CR)

PL이 CR 보다 좋은 점은 data augmentation이 필요하지 않으며 대부분의 도메인에 적용될 수 있다는 점입니다.

하지만 최근 CR 방법론은 SSL benchmark dataset에서 PL 보다 좋은 성능을 보이고 있습니다.

그러나 이 연구는 PL을 support하는 주장을 펼치며, PL이 CR 만큼 중요하다는 입장을 보이고 있습니다.

앞서, pseudo-labeling 방법론은 high confidence를 가지는 unlabeled samples를 선별해 decision boundary를 low density regions으로 옮긴다고 했습니다. 선별된 predictions 중 많은 것들이 "poor calibration"로 인해 틀린 예측이었습니다. 이 연구에서는 이러한 nosiy training과 poor generalization에 문제를 제기하고, output prediction uncertainty와 calibration 같의 관계를 연구하므로써 해결책을 제시합니다. Low uncertainty를 갖는 prediction을 선택하는 것이 poor calibration의 효과를 감소시키고, 개선된 generalization을 보였다고 합니다.



