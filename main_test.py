import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
seq_length=30
def create_sequences_and_labels(data, seq_length):
    sequences = []
    labels = []
    for i in range(len(data) - seq_length - 10):
        sequences.append(data[i:i + seq_length])
        next_values = data[i + seq_length:i + seq_length + 10, 3]  # 종가
        initial_price = next_values[0]
        max_price = np.max(next_values)
        min_price = np.min(next_values)
        if max_price >= initial_price * 1.01 and max_price <= initial_price * 0.99:
            labels.append(0)  # 횡보
        elif max_price >= initial_price * 1.01:
            labels.append(1)  # 상승
        else:
            labels.append(2)  # 하락
    return np.array(sequences), np.array(labels)


# 새로운 csv 파일에서 데이터 읽어오기
bitcoin_test_df = pd.read_csv('bit_origin_test.csv')
nasdaq_test_df = pd.read_csv('nas_origin_test.csv')

# 시간 정보를 datetime으로 변환
bitcoin_test_df['time'] = pd.to_datetime(bitcoin_test_df['time'])
nasdaq_test_df['time'] = pd.to_datetime(nasdaq_test_df['time'])

# 두 데이터 프레임을 시간에 따라 결합
merged_test_df = pd.merge(bitcoin_test_df, nasdaq_test_df, on='time', suffixes=('_btc', '_nasdaq'))

# Normalization
scaler = MinMaxScaler()
scaled_test_data = scaler.transform(merged_test_df.iloc[:, 1:])

# 테스트 데이터를 시퀀스로 변환
test_sequences, test_labels = create_sequences_and_labels(scaled_test_data, seq_length)

# 텐서로 변환
test_sequences_tensor = torch.Tensor(test_sequences)
test_labels_tensor = torch.LongTensor(test_labels)

# 모델 가중치 불러오기
model.load_state_dict(torch.load('model_weights.pth'))

# 모델을 평가 모드로 설정
model.eval()

# 예측 수행
with torch.no_grad():
    outputs = model(test_sequences_tensor)

# 예측값에서 가장 높은 확률을 갖는 레이블 찾기
_, predicted = torch.max(outputs.data, 1)

# 정확도 계산
correct = (predicted == test_labels_tensor).sum().item()
accuracy = correct / len(test_labels_tensor)

print(f'Test Accuracy: {accuracy * 100}%')
