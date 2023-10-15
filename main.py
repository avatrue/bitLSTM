import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# 데이터 로드 및 전처리
def load_and_preprocess_data():
    bitcoin_df = pd.read_csv('bit_learn_7_2.csv')
    nasdaq_df = pd.read_csv('nas_learn_7_2.csv')

    bitcoin_df['time'] = pd.to_datetime(bitcoin_df['time'])
    nasdaq_df['time'] = pd.to_datetime(nasdaq_df['time'])

    merged_df = pd.merge(bitcoin_df, nasdaq_df, on='time', suffixes=('_btc', '_nasdaq'))

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(merged_df.iloc[:, 1:])

    return scaled_data, scaler

# 시퀀스 및 레이블 생성
def create_sequences_and_labels(data, seq_length):
    sequences = []
    labels = []
    for i in range(len(data) - seq_length - 10):
        sequences.append(data[i:i + seq_length])
        next_values = data[i + seq_length:i + seq_length + 10, 3]
        initial_price = next_values[0]
        label = 0
        for next_price in next_values:
            if next_price >= initial_price * 1.01:
                label = 1
                break
            elif next_price <= initial_price * 0.99:
                label = 2
                break
        labels.append(label)

    return np.array(sequences), np.array(labels)

# LSTM 모델 정의
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


# 모델 학습
def train_model(model, criterion, optimizer, dataloader, num_epochs):
    loss_values = []

    for epoch in range(num_epochs):
        total_loss = 0
        for sequences_batch, labels_batch in dataloader:
            outputs = model(sequences_batch)
            loss = criterion(outputs, labels_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        loss_values.append(avg_loss)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

    return loss_values






"""테스트 데이터 로딩 및 전처리를 위한 함수"""
def load_test_data(test_file_names, seq_length, scaler):

    # 데이터 파일에서 테스트 데이터 로드
    bitcoin_test_df = pd.read_csv(test_file_names[0])
    nasdaq_test_df = pd.read_csv(test_file_names[1])

    # 시간 정보 변환
    bitcoin_test_df['time'] = pd.to_datetime(bitcoin_test_df['time'])
    nasdaq_test_df['time'] = pd.to_datetime(nasdaq_test_df['time'])

    # 데이터 병합
    merged_test_df = pd.merge(bitcoin_test_df, nasdaq_test_df, on='time', suffixes=('_btc', '_nasdaq'))

    # 정규화
    scaled_test_data = scaler.transform(merged_test_df.iloc[:, 1:])

    # 시퀀스 및 라벨 생성
    test_sequences, test_labels = create_sequences_and_labels(scaled_test_data, seq_length)

    # 텐서로 변환
    test_sequences_tensor = torch.Tensor(test_sequences)
    test_labels_tensor = torch.LongTensor(test_labels)

    return test_sequences_tensor, test_labels_tensor

"""모델 평가를 위한 함수"""
def evaluate(model, test_sequences_tensor, test_labels_tensor, loss_function, hidden_size, seq_length, loss_values):

    model.eval()  # 모델을 평가 모드로 설정
    correct_predictions = [0, 0, 0]
    total_predictions = [0, 0, 0]

    actuals = []
    losses = []

    with torch.no_grad():
        for i in range(len(test_sequences_tensor)):
            test_sequence = test_sequences_tensor[i].unsqueeze(0)  # 배치 차원 추가
            output = model(test_sequence)

            _, predicted = torch.max(output.data, 1)  # 확률이 가장 높은 레이블 가져오기
            actual_label = test_labels_tensor[i].item()

            print(f'Test Data {i+1}: Predicted Label: {predicted.item()}, Actual Label: {actual_label}')

            # 예측 결과 계산
            total_predictions[predicted.item()] += 1
            if predicted.item() == actual_label:
                correct_predictions[predicted.item()] += 1
            actuals.append(actual_label)

            # 손실 계산
            loss = loss_function(output, test_labels_tensor[i].unsqueeze(0))
            losses.append(loss.item())

    # 각 라벨에 대한 예측 횟수와 맞춘 비율 출력
    for i in range(3):
        if total_predictions[i] > 0:
            accuracy = correct_predictions[i] / total_predictions[i] * 100
        else:
            accuracy = 0
        print(f'Label {i}: Total Predictions: {total_predictions[i]}, Correct Predictions: {correct_predictions[i]}, Accuracy: {accuracy:.2f}%')

    print(f'hiddensize: {hidden_size} seq_length: {seq_length}')
    print(f'Final Loss: {sum(losses) / len(losses):.4f}')  # 모든 테스트 데이터에 대한 평균 손실 출력




def main():
    # 데이터 로드 및 전처리
    scaled_data, scaler = load_and_preprocess_data()

    seq_length = 50
    sequences, labels = create_sequences_and_labels(scaled_data, seq_length)

    sequences_tensor = torch.Tensor(sequences)
    labels_tensor = torch.LongTensor(labels)

    batch_size = 64
    dataset = TensorDataset(sequences_tensor, labels_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 모델 설정
    input_size = scaled_data.shape[1]
    hidden_size = 64
    num_layers = 2
    num_classes = 3

    model = LSTMModel(input_size, hidden_size, num_layers, num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    # 모델 학습
    num_epochs = 1
    loss_values = train_model(model, criterion, optimizer, dataloader, num_epochs)

    # 결과 시각화
    plt.plot(loss_values)
    plt.title('Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

    # 모델 가중치 저장
    torch.save(model.state_dict(), 'model_weights.pth')

    # 테스트 데이터 로드
    test_file_names = ['bit_test_3.csv', 'nas_test_3.csv']
    test_sequences_tensor, test_labels_tensor = load_test_data(test_file_names, seq_length, scaler)

    # 모델 가중치 불러오기
    model.load_state_dict(torch.load('model_weights_12.pth'))

    final_loss = evaluate(model, test_sequences_tensor, test_labels_tensor, criterion, hidden_size, seq_length, loss_values)



if __name__ == "__main__":
    main()
