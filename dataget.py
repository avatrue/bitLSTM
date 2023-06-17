import csv

filename = "15bit.csv"

with open(filename, 'r') as file:
    reader = csv.reader(file)
    next(reader)  # 첫 번째 줄(헤더)은 건너뜁니다.

    for row in reader:
        time, open_price, high_price, low_price, close_price, volume = row[:6]
        break  # 첫 번째 줄만 읽어오고 반복문을 종료합니다.

print("time:", time)
print("open:", open_price)
print("high:", high_price)
print("low:", low_price)
print("close:", close_price)
print("volume:", volume)
