import scipy.io
import pandas as pd
import os
import glob

def batch_convert_mat_to_csv(input_folder, output_folder):
    """
    批量将一个文件夹中的所有MAT文件转换为CSV文件
    
    参数:
        input_folder (str): 包含MAT文件的输入文件夹路径
        output_folder (str): 保存CSV文件的输出文件夹路径
    """
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"创建输出文件夹: {output_folder}")
    
    # 获取所有MAT文件
    mat_files = glob.glob(os.path.join(input_folder, "*.mat"))
    
    if not mat_files:
        print(f"在文件夹 '{input_folder}' 中未找到任何.mat文件")
        return
    
    print(f"找到 {len(mat_files)} 个MAT文件，开始转换...")
    
    # 处理每个MAT文件
    for mat_file in mat_files:
        try:
            # 提取文件名（不含扩展名）
            base_name = os.path.splitext(os.path.basename(mat_file))[0]
            print(f"\n正在处理文件: {base_name}.mat")
            
            # 加载MAT文件
            mat_data = scipy.io.loadmat(mat_file)
            
            # 处理每个变量
            variables_processed = 0
            for var_name, var_data in mat_data.items():
                # 跳过系统变量
                if var_name.startswith('__'):
                    continue
                
                # 处理不同类型的变量
                if var_data.ndim == 1:  # 一维时间序列数据
                    df = pd.DataFrame(var_data, columns=[var_name])
                elif var_data.ndim == 2 and var_data.shape[0] == 1 and var_data.shape[1] == 1:  # 单值变量
                    df = pd.DataFrame([var_data[0, 0]], columns=[var_name])
                else:  # 其他情况
                    df = pd.DataFrame(var_data)
                
                # 生成输出文件名并保存
                output_file = os.path.join(output_folder, f"{base_name}_{var_name}.csv")
                df.to_csv(output_file, index=False)
                print(f"  → 已转换变量 '{var_name}' 为: {base_name}_{var_name}.csv")
                variables_processed += 1
            
            if variables_processed == 0:
                print(f"  警告: 文件 {base_name}.mat 中未找到有效变量")
            else:
                print(f"  成功转换 {variables_processed} 个变量")
                
        except Exception as e:
            print(f"  处理文件 {mat_file} 时出错: {e}")
    
    print(f"\n批量转换完成! 所有CSV文件已保存到: {output_folder}")

# 使用示例
if __name__ == "__main__":
    # 请修改为您的实际路径
    input_folder = "E:\CRWU_code\CaseWesternReserveUniversityData-master"  # 包含所有MAT文件的文件夹
    output_folder = "E:\CRWU_code\CRWU_CSV"  # 保存CSV文件的文件夹
    
    # 执行批量转换
    batch_convert_mat_to_csv(input_folder, output_folder)