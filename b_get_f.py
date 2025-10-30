import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----------------------
# 1. 读取单列CSV数据
# ----------------------
# 假设CSV只有一列信号值（列名可能为空，用0索引读取）
df = pd.read_csv("E:\CRWU_code\CRWU_CSV\\12k_Drive_End_B007_0_118_X118_DE_time.csv", header=None)  # 替换为你的CSV路径
y = df[0].values  # 提取单列信号值
n = len(y)  # 信号长度


# ----------------------
# 2. 生成时间序列（关键：需要采样频率）
# ----------------------
# 采样频率（fs）：需用户根据实际采集设备/场景填写（例如：1000 Hz 表示每秒采1000个点）
# 若不确定，需确认设备参数（如传感器采样率）或数据总时长（fs = 总点数 / 总时长）
fs = 1000  # 示例：1000 Hz，根据实际情况修改！！！

# 生成时间序列：从0开始，间隔1/fs，共n个点
t = np.arange(n) / fs  # 时间单位：秒


# ----------------------
# 3. 信号预处理（同之前）
# ----------------------
# 去除直流分量（减去均值）
y_processed = y - np.mean(y)

# 加窗减少频谱泄露（汉宁窗）
window = np.hanning(n)
y_windowed = y_processed * window


# ----------------------
# 4. FFT计算与频域分析
# ----------------------
# 执行FFT并计算振幅谱
yf = np.fft.fft(y_windowed)
amplitude = 2 * np.abs(yf[:n//2]) / n  # 振幅归一化

# 计算频率轴（0 ~ fs/2，奈奎斯特频率）
freq = fs * np.arange(n//2) / n


# ----------------------
# 5. 提取主要频率
# ----------------------
# 忽略接近0的低频干扰（可调整min_freq）
min_freq = 0.1  # 最小关注频率
valid_idx = np.where(freq > min_freq)[0]

if len(valid_idx) == 0:
    main_freq = 0.0
else:
    max_amp_idx = valid_idx[np.argmax(amplitude[valid_idx])]
    main_freq = freq[max_amp_idx]

print(f"采样频率：{fs} Hz")
print(f"主要频率成分：{main_freq:.2f} Hz")


# ----------------------
# 6. 可视化
# ----------------------
plt.figure(figsize=(12, 6))

# 时域信号
plt.subplot(2, 1, 1)
plt.plot(t, y, label="原始信号")
plt.xlabel("时间（秒）")
plt.ylabel("信号值")
plt.legend()

# 频域振幅谱
plt.subplot(2, 1, 2)
plt.plot(freq, amplitude, label="振幅谱")
plt.axvline(x=main_freq, color='r', linestyle='--', label=f"主要频率：{main_freq:.2f} Hz")
plt.xlabel("频率（Hz）")
plt.ylabel("振幅")
plt.xlim(0, fs/5)  # 限制显示范围，避免高频噪声干扰
plt.legend()

plt.tight_layout()
plt.show()