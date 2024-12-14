import pandas as pd

# Đọc tệp cleaned_labeled_data.csv
df = pd.read_csv('cleaned_labeled_data.csv')

# Đảm bảo các cột cần thiết tồn tại
required_columns = {'Cleaned_Comment', 'Label'}
if not required_columns.issubset(df.columns):
    raise ValueError(f'Tệp CSV thiếu các cột cần thiết: {required_columns - set(df.columns)}')

# Xác định số mẫu của nhãn NEU
neu_count = df[df['Label'] == 'NEU'].shape[0]

# Số mẫu cho mỗi nhãn trong tập kiểm thử (test) = số mẫu NEU / 5 (làm tròn xuống)
num_test_samples_per_label = max(1, neu_count // 5)  # Đảm bảo ít nhất là 1 mẫu

# Lấy mẫu từ mỗi nhãn (POS, NEG, NEU) với số lượng đã xác định
test_df = df.groupby('Label', group_keys=False).apply(
    lambda x: x.sample(min(len(x), num_test_samples_per_label), random_state=42, ignore_index=True)
)

# Tạo tệp train.csv với dữ liệu còn lại (ngoại trừ những mẫu đã chọn cho test.csv)
train_df = df.drop(test_df.index)

# Xuất tệp test.csv và train.csv
test_df.to_csv('test.csv', index=False)
train_df.to_csv('train.csv', index=False)

print(f'Số mẫu kiểm thử cho mỗi nhãn (POS, NEG, NEU): {num_test_samples_per_label}')
print(f'Số mẫu trong tập train.csv: {train_df.shape[0]}')
print(f'Số mẫu trong tập test.csv: {test_df.shape[0]}')
