seed_value= 42
import os
os.environ['PYTHONHASHSEED']=str(seed_value)
import random
random.seed(seed_value)
import numpy as np
np.random.seed(seed_value)
import tensorflow as tf
tf.random.set_seed(seed_value)


import numpy as np
import keras
from keras import layers
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import model_from_json
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
from keras.utils import plot_model
import seaborn
import process as pre_proc
import matplotlib.pyplot as plt
from keras.optimizers import SGD, Adam, RMSprop

### enable/disable attention ###
ENABLE_ATTENTION = True

def create_model(units=64, optimizer='adam', learning_rate=0.01):
    input = keras.Input(shape=(pre_proc.N_FRAMES, pre_proc.N_FEATURES))#这里定义了模型的输入层，其形状为 (N_FRAMES, N_FEATURES)，这个形状反映了输入特征的维度。
    if MODEL == "Attention_BLSTM":#这个条件语句根据指定的 MODEL 参数选择不同的模型架构。在这个条件下，选择的是 Attention BLSTM 模型。
        states, forward_h, _, backward_h, _ = layers.Bidirectional(layers.LSTM(units, return_sequences=True, return_state=True))(input)#这段代码创建了一个双向 LSTM 层，并且将其应用于输入数据。return_sequences=True 表示该层将返回每个时间步的输出，而 return_state=True 表示该层将返回前向和后向 LSTM 的状态信息。
        last_state = layers.Concatenate()([forward_h, backward_h])#这里将前向和后向 LSTM 的状态连接起来，以形成最终的状态向量。
        hidden = layers.Dense(units, activation="tanh", use_bias=False,kernel_initializer=keras.initializers.RandomNormal(mean=0., stddev=1.))(states)#创建一个全连接层，用于处理 LSTM 层的输出。这个层将输出转换为具有 tanh 激活函数的隐藏层。
        hidden = layers.Dropout(0.5)(hidden)  # 添加 Dropout 层
        hidden = layers.Dropout(0.5)(hidden)
        hidden = layers.Dropout(0.5)(hidden)
        out = layers.Dense(1, activation='linear', use_bias=False,kernel_initializer=keras.initializers.RandomNormal(mean=0., stddev=1.))(hidden)#这里创建了一个线性激活函数的输出层，其输出与隐藏层的单元数一致。
        flat = layers.Flatten()(out)
        energy = layers.Lambda(lambda x:x/np.sqrt(units))(flat)#这一行计算输出层的能量。将层 flat 的输出除以 units 的平方根。这个操作是一种常见的能量标准化方式。
        normalize = layers.Softmax()
        normalize._init_set_name("alpha")
        alpha = normalize(energy)#energy 通过 softmax 激活函数。这个操作将能量值归一化为概率分布，确保它们的总和为1。alpha 表示注意力权重。
        context_vector = layers.Dot(axes=1)([states, alpha])
        context_vector = layers.Concatenate()([context_vector, last_state])#将注意力权重应用到 LSTM 的状态上，并将最终的状态向量与上一步中计算的最终状态向量连接起来。
    elif MODEL == "BLSTM":
        context_vector = layers.Bidirectional(layers.LSTM(units, return_sequences=False))(input)
    else:
        raise Exception("Unknown model architecture!")
    pred = layers.Dense(pre_proc.N_EMOTIONS, activation="softmax")(context_vector)#最后，根据上一步的结果生成模型的最终预测。
    model = keras.Model(inputs=[input], outputs=[pred])#创建 Keras 模型对象，指定输入和输出。
    model._init_set_name(MODEL)#为模型命名。
    # 根据参数选择不同的优化器
    if optimizer == 'sgd':
        optimizer = SGD(lr=0.01)
    elif optimizer == 'adam':
        optimizer = Adam(lr=0.001)
    elif optimizer == 'rmsprop':
        optimizer = RMSprop(lr=0.001)
    else:
        raise ValueError("Unknown optimizer")
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    print(str(model.summary()))#打印模型的摘要信息。
    return model#返回创建的模型对象。


def train_and_test_model(model):#这是一个函数定义，它接受一个深度学习模型作为输入。
    X_train, X_test, y_train, y_test = pre_proc.get_train_test()#这一行从名为pre_proc的模块中获取训练和测试数据集。
    print(X_train, X_test, y_train, y_test)
    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])#这一行编译了模型，使用分类交叉熵作为损失函数，Adam优化器进行优化，并且衡量指标为准确率。
    plot_model(model, MODEL+"_model.png", show_shapes=True)#这一行生成模型的架构图，并保存为一个PNG文件。
    best_weights_file = MODEL+"_weights.h5"#这一行定义了用于保存最佳权重的文件名。
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=2, patience=10)
    mc = ModelCheckpoint(best_weights_file, monitor='val_loss', mode='min', verbose=2,
                         save_best_only=True)#这两行定义了早期停止和模型检查点的回调函数，用于在验证集上监视验证损失，以便提前停止训练并保存最佳模型权重。
    history = model.fit(X_train, y_train,validation_data=(X_test, y_test),epochs=10,batch_size=32,callbacks=[es, mc],verbose=2)#这一行训练模型，并且记录训练历史。
    save(model)#保存模型。
    # model testing
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    if MODEL == "Attention_BLSTM":
        plt.title('model accuracy - BLSTM with attention')
    else:
        plt.title('model accuracy - BLSTM without attention')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'text'], loc='upper left')
    plt.savefig(MODEL+"_accuracy.png")
    plt.gcf().clear()  # clear
    # loss on validation
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    if MODEL == "Attention_BLSTM":
        plt.title('model loss - BLSTM with attention')
    else:
        plt.title('model loss - BLSTM without attention')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'text'], loc='upper left')
    plt.savefig(MODEL+"_loss.png")
    plt.gcf().clear()  # clear
    # test acc and loss
    model.load_weights(best_weights_file) # load the best saved model这一行加载最佳权重。
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    test_metrics = model.evaluate(X_test, y_test, batch_size=32)#评估模型在测试集上的性能。
    print("\n%s: %.2f%%" % ("test " + model.metrics_names[1], test_metrics[1] * 100))
    print("%s: %.2f" % ("test " + model.metrics_names[0], test_metrics[0]))
    print("test accuracy: " + str(format(test_metrics[1], '.3f')) + "\n")
    print("test loss: " + str(format(test_metrics[0], '.3f')) + "\n")

    # 计算模型的精确率、召回率和F1分数
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    precision = precision_score(y_true_classes, y_pred_classes, average='weighted')
    recall = recall_score(y_true_classes, y_pred_classes, average='weighted')
    f1 = f1_score(y_true_classes, y_pred_classes, average='weighted')

    print("Precision: %.2f" % precision)
    print("Recall: %.2f" % recall)
    print("F1-score: %.2f" % f1)

    # test acc and loss per class
    real_class = np.argmax(y_test, axis=1)
    pred_class_probs = model.predict(X_test)
    pred_class = np.argmax(pred_class_probs, axis=1)#生成测试集上的真实类别和预测类别，并计算混淆矩阵和分类报告。

    report = classification_report(real_class, pred_class)
    print("classification report:\n" + str(report) + "\n")
    cm = confusion_matrix(real_class, pred_class)
    print("confusion_matrix:\n" + str(cm) + "\n")
    data = np.array([value for value in cm.flatten()]).reshape(6,6)
    if MODEL == "Attention_BLSTM":
        plt.title('BLSTM with attention')
    else:
        plt.title('BLSTM without attention')
    seaborn.heatmap(cm, xticklabels=pre_proc.emo_labels_en, yticklabels=pre_proc.emo_labels_en, annot=data, cmap="Reds")
    plt.savefig(MODEL+"_conf_matrix.png")


def visualize_attention(model):#函数定义，它接受一个模型作为输入。
    best_weights_file = MODEL + "_weights.h5"
    model.load_weights(best_weights_file)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])#这几行加载了训练过程中保存的最佳权重，并编译了模型。
    _, X_test, _, _ = pre_proc.get_train_test()#从pre_proc模块中获取测试数据集。
    predictions = model.predict(X_test)
    labels = np.argmax(predictions, axis=1)#用于进行模型的预测，并提取预测标签。
    # inspect attention weigths
    attention = model.get_layer(name="alpha")
    weigth_model = keras.Model(inputs=model.input, outputs=attention.output)
    attention_weights = weigth_model.predict(X_test)#获取模型中的注意力权重，并将其输出为一个新的模型，以便提取注意力权重。
    d = {}
    for w, l in zip(attention_weights, labels):
        if l not in d:
            d[l] = w
        else:
            d[l] += w #对每个类别的注意力权重进行汇总。
    data = []
    for x, y in d.items():
        norm_w = y / np.sum(y)
        data.append(norm_w)#对每个类别的注意力权重进行归一化处理。
    #reshape and trim
    bins = 10
    bin_c = pre_proc.N_FRAMES//bins
    trim = pre_proc.N_FRAMES%bins
    data = np.asarray(data).reshape(pre_proc.N_EMOTIONS, pre_proc.N_FRAMES)[:, trim:]
    data = np.sum(data.reshape([6, bins, bin_c]), axis=2).reshape(pre_proc.N_EMOTIONS,bins)#将注意力权重数据重新排列，以便进行可视化。
    plt.clf()
    seaborn.heatmap(data, yticklabels=pre_proc.emo_labels_en, cmap="Reds")
    plt.savefig("visualize_attention.png")
    print(data)


def load():#函数定义，用于加载模型。
    with open("model.json", 'r') as f:
        model = model_from_json(f.read())#打开一个名为"model.json"的文件，该文件包含了模型的结构信息，并使用model_from_json函数将其加载为一个模型对象
    best_weights_file = MODEL + "_weights.h5"#定义了用于加载模型权重的文件名。
    # Load weights into the new model
    model.load_weights(best_weights_file)#加载了之前训练保存的模型权重。
    return model


def save(model):#函数定义，用于保存模型。
    model_json = model.to_json()#将模型转换为JSON格式。
    with open(MODEL+"_model.json", "w") as json_file:
        json_file.write(model_json)#这一行将模型的JSON表示写入到文件中，文件名由变量MODEL和后缀"_model.json"组成。
    print("model saved")




######### SPEECH EMOTION RECOGNITION #########

# 1) feature extraction
pre_proc.feature_extraction()#这一行调用了一个名为feature_extraction的函数，该函数可能用于特征提取，但在提供的代码中并未提供具体实现。

# 2) select model
if ENABLE_ATTENTION:
    MODEL = "Attention_BLSTM"
else:
    MODEL = "BLSTM"
#这段代码根据一个名为ENABLE_ATTENTION的布尔值来选择使用哪种模型，如果ENABLE_ATTENTION为True，则选择"Attention_BLSTM"模型，否则选择"BLSTM"模型。
# 3) create model
model = create_model()
#这一行调用了一个名为create_model的函数，用于创建模型，并将创建的模型赋给变量model。
# 4) train and test model
train_and_test_model(model)
#这一行调用了一个名为train_and_test_model的函数，用于训练和测试模型。它接受一个模型作为参数。
# 5) visualize attention weights
if ENABLE_ATTENTION:
    visualize_attention(model)
    #这段代码根据ENABLE_ATTENTION的值来确定是否可视化注意力权重。如果ENABLE_ATTENTION为True，则调用visualize_attention函数对模型的注意力权重进行可视化。