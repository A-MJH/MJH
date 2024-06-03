
seed_value= 42
import os
os.environ['PYTHONHASHSEED']=str(seed_value)
import random
random.seed(seed_value)
import numpy as np
np.random.seed(seed_value)
import tensorflow as tf
tf.random.set_seed(seed_value)

import math
import librosa
import os
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from sklearn.preprocessing import StandardScaler as std, OneHotEncoder
from imblearn.over_sampling import SMOTE
import pickle

sr=16000
duration=5
frame_length=400
N_FRAMES = math.ceil(sr * duration / frame_length)
N_EMOTIONS=6
N_FEATURES=46
path= "C:/Users/马嘉灏/PycharmProjects/pythonProject3/语音情感识别代码/语音数据2/Crema"
emo_codes = {"ANG": 0, "DIS": 1, "FEA": 2, "HAP": 3, "NEU": 4, "SAD": 5}
emo_labels_en = ["anger",  "disgust", "fear", "happiness","neutral", "sadness"]

def get_emotion_label(file_name):
    emo_code = file_name[9:12]  # 从文件名中提取最后一个字符作为情感代码
    return emo_codes[emo_code]  # 返回情感代码对应的情感标签


def feature_extraction():#函数定义，函数名为 feature_extraction，用于执行音频特征提取。
    wavs = []#创建一个空列表 wavs，用于存储加载的音频数据
    # load 16 kHz resampled files
    for file in os.listdir(path):#使用 os.listdir(path) 遍历给定路径 path 下的所有文件
        y, _ = librosa.load(path + "/" + file, sr=sr, mono=True, duration=duration)#使用 librosa.load() 函数加载音频文件，其中 sr 是采样率，mono 是指定是否将信号转换为单声道，duration 是指定加载的音频片段持续时间。加载的音频数据存储在变量 y 中。
        wavs.append(y)#将加载的音频数据 y 添加到 wavs 列表中
    # pad to fixed length (zero, 'pre')
    wavs_padded = pad_sequences(wavs, maxlen=sr * duration, dtype="float32")#使用 pad_sequences() 函数将音频数据填充到指定的最大长度。这是为了确保所有的音频片段具有相同的长度。
    features = []  # (N_SAMPLES, N_FRAMES, N_FEATURES)
    emotions = []  # 创建两个空列表 features 和 emotions，用于存储特征和情感标签。
    for y, name in zip(wavs_padded, os.listdir(path)):  # 使用 zip() 函数同时迭代音频数据 wavs_padded 和文件名列表，其中 y 是音频数据，name 是文件名。
        frames = []  # 创建一个空列表 frames，用于存储每个音频片段的特征帧。
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=frame_length)[0]  # 提取音频的频谱中心特征。
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=frame_length)[0]  # 频谱对比度
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=frame_length)[0]  # 频谱带宽
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=frame_length)[0]  # 频谱滚降
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y, hop_length=frame_length)[0]  # 过零率
        S, phase = librosa.magphase(librosa.stft(y=y, hop_length=frame_length))
        rms = librosa.feature.rms(y=y, hop_length=frame_length, S=S)[0]  # RMS
        mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=frame_length)  # MFCC
        mfcc_der = librosa.feature.delta(mfcc)
        for i in range(N_FRAMES):  # 循环迭代每个帧，并将提取的特征存储在 frames 列表中。
            f = []
            f.append(spectral_centroid[i])
            f.append(spectral_contrast[i])
            f.append(spectral_bandwidth[i])
            f.append(spectral_rolloff[i])
            f.append(zero_crossing_rate[i])
            f.append(rms[i])
            for m_coeff in mfcc[:, i]:
                f.append(m_coeff)
            for m_coeff_der in mfcc_der[:, i]:
                f.append(m_coeff_der)
            frames.append(f)
        features.append(frames)
        emotions.append(get_emotion_label(name))  # 将每个音频片段的特征帧添加到 features 列表中，并获取相应的情感标签并添加到 emotions 列表中。
    features = np.array(features)
    emotions = np.array(emotions)  # 将列表转换为 NumPy 数组。
    print(str(features.shape))  # 打印特征数组的形状。
    pickle.dump(features, open("features.p", "wb"))
    pickle.dump(emotions, open("emotions.p", "wb"))  # 使用 pickle 序列化特征数组和情感标签数组，并将它们保存到文件中。

def get_train_test(test_samples_per_emotion=60):#这是一个函数定义，名为 get_train_test，它接受一个参数 test_samples_per_emotion，用于指定每个情感类别在测试集中的样本数量，默认为 20。
    features = pickle.load(open("features.p", "rb"))
    emotions = pickle.load(open("emotions.p", "rb"))#这两行代码加载了特征数组和情感标签数组，这些数组是从之前保存的文件中反序列化而来。
    # flatten
    N_SAMPLES = len(features)
    features.shape = (N_SAMPLES, N_FRAMES * N_FEATURES)#这两行代码获取了特征数组的样本数量，并将特征数组的形状调整为 (N_SAMPLES, N_FRAMES * N_FEATURES)，即每个样本的特征都被扁平化成一个一维数组。
    # standardize data
    scaler = std()
    features = scaler.fit_transform(features)#创建了一个标准化（StandardScaler）对象 scaler，并使用它对特征进行标准化处理，使得特征的均值为 0，标准差为 1。
    # shuffle
    perm = np.random.permutation(N_SAMPLES)
    features = features[perm]
    emotions = emotions[perm]#使用 np.random.permutation() 对样本进行随机排列，然后将特征数组和情感标签数组按照相同的顺序重新排列。
    # get balanced test set of real samples
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    count_test = np.zeros(N_EMOTIONS)#创建了四个空列表 X_train, y_train, X_test, y_test，用于存储训练集和测试集的特征及其对应的情感标签。同时创建了一个长度为 N_EMOTIONS 的零数组 count_test，用于计算每个情感类别在测试集中的样本数量。
    for f,e in zip(features, emotions):
        if count_test[e] < test_samples_per_emotion:
            X_test.append(f)
            y_test.append(e)
            count_test[e]+=1
        else:
            X_train.append(f)
            y_train.append(e)
    #遍历特征数组和情感标签数组，根据 test_samples_per_emotion 确保每个情感类别在测试集中的样本数量不超过指定值。
    # 如果某个情感类别在测试集中的样本数量未达到指定值，则将其样本添加到测试集中；否则，将其样本添加到训练集中。
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test)#将列表转换为 NumPy 数组。
    y_train_original=y_train
    # balance train classes
    sm = SMOTE()#使用 SMOTE（Synthetic Minority Over-sampling Technique）算法对训练集进行过采样，以解决类别不平衡问题。
    X_train, y_train = sm.fit_resample(X_train, y_train)#将训练集和测试集的形状重新调整为 (样本数量, 帧数, 特征数)，以便后续的模型训练。
    y_train_resampled=y_train
    # analyze emotion distribution
    # restore 3D shape
    X_train.shape = (len(X_train), N_FRAMES, N_FEATURES)
    X_test.shape = (len(X_test), N_FRAMES, N_FEATURES)
    # encode labels in one-hot vectors
    encoder = OneHotEncoder(sparse=False, drop=None)
    #encoder = enc(sparse=False)#使用 OneHotEncoder 对情感标签进行独热编码，以便用于多类别分类任务。
    y_train = np.array(y_train).reshape(-1, 1)
    y_train = encoder.fit_transform(y_train)
    y_test = np.array(y_test).reshape(-1, 1)
    y_test = encoder.fit_transform(y_test)
    # 计算过采样前各情绪类别的样本数量
    count_before = [np.sum(y_train_original == i) for i in range(len(emo_labels_en))]

    # 计算过采样后各情绪类别的样本数量
    count_after = [np.sum(y_train_resampled == i) for i in range(len(emo_labels_en))]

    # 绘制柱形图
    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.35
    index = np.arange(len(emo_labels_en))

    bar1 = ax.bar(index, count_before, bar_width, label='Before Oversampling')
    bar2 = ax.bar(index + bar_width, count_after, bar_width, label='After Oversampling')

    ax.set_xlabel('Emotions')
    ax.set_ylabel('Number of Samples')
    ax.set_title('Emotion Distribution Before and After Oversampling')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(emo_labels_en)
    ax.legend()

    plt.show()
    return X_train, X_test, y_train, y_test#返回处理后的训练集和测试集数据。

def plot_features(features):
    num_features = features.shape[2]  # 获取特征数量
    plt.figure(figsize=(12, 8*num_features))  # 设置图像大小
    for i in range(num_features):
        plt.subplot(num_features, 1, i+1)
        librosa.display.specshow(features[:, :, i], x_axis='time', sr=sr, hop_length=frame_length)
        plt.colorbar(format='%+2.0f dB')
        plt.title('Feature {}'.format(i+1))
    plt.tight_layout()
    plt.show()

# 调用 feature_extraction 函数来执行特征提取并绘制特征图像
feature_extraction()


def visualize_features(X_train, X_test):
    # Visualize some features from X_train and X_test
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(X_train[0].flatten(), label='Train Sample 1')
    plt.plot(X_test[0].flatten(), label='Test Sample 1')
    plt.title('Feature Visualization')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(X_train[1].flatten(), label='Train Sample 2')
    plt.plot(X_test[1].flatten(), label='Test Sample 2')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Call the function to get train and test data
X_train, X_test, y_train, y_test = get_train_test()

# 调用 visualize_features 函数生成归一化后的特征可视化图
visualize_features(X_train, X_test)


def plot_one_hot_encoding(y_encoded, title):
    plt.figure(figsize=(10, 6))
    plt.imshow(y_encoded.T, aspect='auto', cmap='hot', interpolation='nearest')
    plt.title(title)
    plt.xlabel('Sample Index')
    plt.ylabel('Emotion Class')
    plt.colorbar(label='Presence')
    plt.show()

# 使用示例
X_train, X_test, y_train, y_test = get_train_test()  # 获取训练集和测试集
plot_one_hot_encoding(y_train, 'One-Hot Encoded Representation (Train)')
plot_one_hot_encoding(y_test, 'One-Hot Encoded Representation (Test)')



# 加载特征数据
features = pickle.load(open("features.p", "rb"))

# 调用 plot_features 函数绘制特征图像
plot_features(features)

def plot_features(features):
    num_features = features.shape[2]  # 获取特征数量
    plt.figure(figsize=(12, 8*num_features))  # 设置图像大小
    for i in range(min(2, num_features)):  # 只绘制前两个特征
        plt.subplot(2, 1, i+1)
        plt.plot(features[:, :, i].flatten())  # 绘制第 i 个特征
        plt.title('Feature {}'.format(i+1))  # 添加特征名称
        plt.xlabel('Time')  # x 轴标签
        plt.ylabel('Value')  # y 轴标签
    plt.tight_layout()
    plt.show()

# 调用 feature_extraction 函数来执行特征提取并绘制特征图像
feature_extraction()
