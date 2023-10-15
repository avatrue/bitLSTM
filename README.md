# bitLSTM
# 주식 가격 예측 LSTM 모델

이 프로젝트는 비트코인과 나스닥 지수의 과거 데이터를 기반으로 미래의 주가 움직임 (상승, 하락, 횡보)을 예측하는 LSTM 딥러닝 모델을 학습시키기 위한 것입니다.

## 설치 및 사용 방법

이 코드를 실행하기 전에, 필요한 라이브러리들을 설치해야 합니다. 필요한 라이브러리들은 다음과 같습니다:

- pandas
- numpy
- torch
- sklearn
- matplotlib


## 데이터 준비

이 스크립트를 실행하기 전에, 로컬 환경에 `bit_learn_7_2.csv`, `nas_learn_7_2.csv`, `bit_test_3.csv`, `nas_test_3.csv` 데이터 파일이 필요합니다.

`bit_learn_7_2.csv` 및 `nas_learn_7_2.csv` 파일은 학습에 사용되며, `bit_test_3.csv` 및 `nas_test_3.csv` 파일은 모델을 평가하기 위한 테스트 데이터로 사용됩니다.
bit_learn_7_2.csv: 비트코인 15분간격 데이터 22년 7월-23년 2월
nas_learn_7_2.csv: 나스닥 선물지수 15분간격 데이터 22년 7월 - 23년 2월

bit_test_3.csv: 비트코인 15분간격 데이터 23년 3월
nas_test_3.csv: 나스닥 선물지수 15분간격 데이터 23년 3월

model_weights.pth 형식 파일에 저장이 되며 현재 model_weights12.pth에 최적화한 결과값이 입력되어 있습니다.
## 스크립트 실행

모든 필요한 라이브러리를 설치한 후, 스크립트를 다음과 같이 실행할 수 있습니다:

python main.py


## 작동 원리

1. **데이터 로드 및 전처리**: 첫 단계에서는 CSV 파일을 로드하고, 필요한 전처리를 진행합니다. 시간 데이터를 파싱하고, 두 데이터 세트를 병합한 후, 스케일링을 진행합니다.
2. **시퀀스 및 레이블 생성**: 학습에 사용될 시퀀스와 레이블을 생성합니다. 이 과정에서 미래 주가 움직임에 따른 레이블을 할당합니다.
3. **LSTM 모델 정의 및 학습**: LSTM 모델을 정의하고, 이전 단계에서 준비된 데이터를 사용하여 모델을 학습시킵니다. model_weights.pth 파일에 저장됩니다
4. **모델 평가**: 테스트 데이터 세트를 사용하여 학습된 모델을 평가합니다. 성능을 계산하고 결과를 출력합니다.

## 결과 및 시각화

학습 과정에서의 손실값 변화를 시각화합니다. 모델의 성능을 분석하기 위해 각 에포크에서 손실값의 변화를 그래프로 표현합니다.
이후 각 테스트데이터의 예측과 실제 결과를 모두 알려줍니다.
이후 각 레이블 별 정확도, hidden size, seq length를 알려줍니다.

### 성능 결과

테스트 데이터를 이용한 모델의 성능은 다음과 같습니다:

레이블 별 정확도:

- Label 0: Total Predictions: 1262, Correct Predictions: 809, Accuracy: 64.10%
- Label 1: Total Predictions: 2154, Correct Predictions: 1518, Accuracy: 70.47%
- Label 2: Total Predictions: 2189, Correct Predictions: 1401, Accuracy: 64.00%

모델 구성:

- Hidden Size: 64
- num_layers = 2
- num_classes = 3
- Sequence Length: 50

최종 손실: 0.7737







