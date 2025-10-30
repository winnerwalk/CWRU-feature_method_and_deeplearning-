import pandas as pd
df = pd.read_csv('./CRWU_CSV/12k_Drive_End_B007_0_118_X118_DE_time.csv')
print(df.head())  # 查看前5行
print(df.shape)   # 查看数据维度
print(df.columns) # 查看列名

import pandas as pd
import os

def process_csv_files():
    # 读取txt文件中的CSV路径
    with open('./c_挑选的数据集.txt', 'r') as f:  # 请替换为您的txt文件路径
        csv_paths = [line.strip() for line in f.readlines() if line.strip()]
    
    all_data = []
    all_labels = []
    
    # 处理每个CSV文件
    for label, csv_path in enumerate(csv_paths):
        # 检查文件是否存在
        if not os.path.exists(csv_path):
            print(f"文件不存在: {csv_path}")
            continue
            
        try:
            # 读取CSV文件（假设数据在第一列）
            df = pd.read_csv(csv_path, header=None)
            data = df.iloc[:, 0].values  # 获取第一列数据
            
            # 按照要求提取数据组
            start_index = 0
            while start_index + 500 <= len(data):
                # 提取500个数据
                data_group = data[start_index:start_index + 500]
                all_data.append(data_group)
                all_labels.append(label)
                
                # 移动200个数据点
                start_index += 200
                
            print(f"处理完成: {csv_path}, 标签: {label}, 提取了 {len([l for l in all_labels if l == label])} 组数据")
            
        except Exception as e:
            print(f"处理文件时出错 {csv_path}: {e}")
            continue
    
    # 创建最终的DataFrame
    result_df = pd.DataFrame({
        'data': all_data,
        'label': all_labels
    })
    
    # 保存到CSV文件
    result_df.to_csv('./dataset/dataset_.csv', index=False)
    print(f"数据保存完成! 总共提取了 {len(all_data)} 组数据")

# 运行处理函数
if __name__ == "__main__":
    process_csv_files()