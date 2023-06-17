import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

# .csv 파일 읽어오기
bitcoin_df = pd.read_csv('bit_learn_7_2.csv')
nasdaq_df = pd.read_csv('nas_learn_7_2.csv')

# 시간 정보를 datetime으로 변환
bitcoin_df['time'] = pd.to_datetime(bitcoin_df['time'])
nasdaq_df['time'] = pd.to_datetime(nasdaq_df['time'])

# 두 데이터 프레임을 시간에 따라 결합
merged_df = pd.merge(bitcoin_df, nasdaq_df, on='time', suffixes=('_btc', '_nasdaq'))

# Normalization
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(merged_df.iloc[:, 1:])  # 시간 제외한 나머지 데이터를 스케일링


# 각각의 데이터를 시퀀스로 분할하고 라벨 생성
# def create_sequences_and_labels(data, seq_length):
#     sequences = []
#     labels = []
#     for i in range(len(data) - seq_length - 10):
#         sequences.append(data[i:i + seq_length])
#         next_values = data[i + seq_length:i + seq_length + 10, 3]  # 종가
#         initial_price = next_values[0]
#         max_price = np.max(next_values)
#         min_price = np.min(next_values)
#         if max_price >= initial_price * 1.01 and max_price <= initial_price * 0.99:
#             labels.append(0)  # 횡보
#         elif max_price >= initial_price * 1.01:
#             labels.append(1)  # 상승
#         else:
#             labels.append(2)  # 하락
#     return np.array(sequences), np.array(labels)
def create_sequences_and_labels(data, seq_length):
    sequences = []
    labels = []
    for i in range(len(data) - seq_length - 10):
        sequences.append(data[i:i + seq_length])
        next_values = data[i + seq_length:i + seq_length + 10, 3]  # 종가
        initial_price = next_values[0]
        label = 0  # 초기값은 횡보로 설정
        for next_price in next_values:
            if next_price >= initial_price * 1.01:  # 1% 이상 상승 시
                label = 1
                break
            elif next_price <= initial_price * 0.99:  # 1% 이상 하락 시
                label = 2
                break
        labels.append(label)
    return np.array(sequences), np.array(labels)


//
seq_length = 50
sequences, labels = create_sequences_and_labels(scaled_data, seq_length)

# Tensor 변환
sequences_tensor = torch.Tensor(sequences)
labels_tensor = torch.LongTensor(labels)

# DataLoader 생성
batch_size = 64
dataset = TensorDataset(sequences_tensor, labels_tensor)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# LSTM 모델
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


# 모델 생성
input_size = scaled_data.shape[1]
hidden_size = 84
num_layers = 2
num_classes = 3
model = LSTMModel(input_size, hidden_size, num_layers, num_classes)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# 학습
num_epochs = 100
loss_values = []  # loss를 저장할 list
for epoch in range(num_epochs):
    total_loss = 0
    total_batches = 0
    for i, (sequences_batch, labels_batch) in enumerate(dataloader):
        outputs = model(sequences_batch)
        loss = criterion(outputs, labels_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_batches += 1

    avg_loss = total_loss / total_batches
    loss_values.append(avg_loss)
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}')

# 모델의 가중치 저장
torch.save(model.state_dict(), 'model_weights_13.pth')

# Loss curve
import matplotlib.pyplot as plt

plt.plot(loss_values)
plt.title('Training Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()


bitcoin_test_df = pd.read_csv('bit_test_3.csv')
nasdaq_test_df = pd.read_csv('nas_test_3.csv')


bitcoin_test_df['time'] = pd.to_datetime(bitcoin_test_df['time'])
nasdaq_test_df['time'] = pd.to_datetime(nasdaq_test_df['time'])

merged_test_df = pd.merge(bitcoin_test_df, nasdaq_test_df, on='time', suffixes=('_btc', '_nasdaq'))

# Normalization

scaled_test_data = scaler.transform(merged_test_df.iloc[:, 1:])


test_sequences, test_labels = create_sequences_and_labels(scaled_test_data, seq_length)

# 텐서로 변환
test_sequences_tensor = torch.Tensor(test_sequences)
test_labels_tensor = torch.LongTensor(test_labels)

# 모델 가중치 불러오기
model.load_state_dict(torch.load('model_weights_13.pth'))

# 평가
model.eval()

#
# with torch.no_grad():
#     outputs = model(test_sequences_tensor)
#
#
# _, predicted = torch.max(outputs.data, 1)
#
#
# correct = (predicted == test_labels_tensor).sum().item()
# accuracy = correct / len(test_labels_tensor)
#
# print(f'Test Accuracy: {accuracy * 100}%')

# 예측 수행
# with torch.no_grad():
#     for i in range(len(test_sequences_tensor)):
#         test_sequence = test_sequences_tensor[i].unsqueeze(0) # 배치 차원 추가
#         output = model(test_sequence)
#
#         # 가장 높은 확률의 레이블
#         _, predicted = torch.max(output.data, 1)
#
#
#         actual_label = test_labels_tensor[i]
#
#         print(f'Test Data {i+1}: Predicted Label: {predicted.item()}, Actual Label: {actual_label.item()}')


# 예측 수행 및 통계 계산
total_predictions = [0, 0, 0]
correct_predictions = [0, 0, 0]

with torch.no_grad():
    for i in range(len(test_sequences_tensor)):
        test_sequence = test_sequences_tensor[i].unsqueeze(0)  # 배치 차원 추가
        output = model(test_sequence)

        # 가장 높은 확률의 레이블
        _, predicted = torch.max(output.data, 1)

        actual_label = test_labels_tensor[i]

        print(f'Test Data {i+1}: Predicted Label: {predicted.item()}, Actual Label: {actual_label.item()}')

        total_predictions[predicted.item()] += 1
        if predicted.item() == actual_label.item():
            correct_predictions[predicted.item()] += 1

# 각 라벨에 대한 예측 횟수와 맞춘 비율 출력
for i in range(3):
    if total_predictions[i] > 0:
        accuracy = correct_predictions[i] / total_predictions[i]
    else:
        accuracy = 0
    print(f'Label {i}: Total Predictions: {total_predictions[i]}, Correct Predictions: {correct_predictions[i]}, Accuracy: {accuracy * 100}%')
print(f'hiddensize:{hidden_size} seq_length:{seq_length}')
print(f'Final Loss: {loss_values[-1]:.4f}')