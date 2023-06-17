import pandas as pd





def get_data_size(file_path):
    # CSV 파일을 데이터프레임으로 읽어옵니다.
    df = pd.read_csv(file_path)

    # 데이터프레임의 크기를 구합니다.
    num_rows, num_columns = df.shape

    # 데이터프레임의 크기를 출력합니다.
    print("Number of rows:", num_rows)
    print("Number of columns:", num_columns)


# 파일 경로 설정
file_path = "nas" \
            "_test_3.csv"

# 데이터 크기 구하기
get_data_size(file_path)
