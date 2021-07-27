from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Dense
from keras import Model
from keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.utils.np_utils import to_categorical
from encoding_module import *

def create_training_data(segment_length=SEGMENT_LENGTH, corpus_path=CORPUS_PATH):
    """为RNN模型创建训练数据。

    :param segment_length (int): 输入序列的长度
    :param corpus_path (str): 语料库的路径

    :return input_notes (ndarray): 训练集的输入序列集合
    :return output_notes (ndarray): 训练集的预测目标集合
    """

    # 加载语料库
    with open(corpus_path, "rb") as filepath:
        corpus = pickle.load(filepath)

    # 训练集的输入序列和预测目标
    input_notes = []
    output_notes = []

    # 得到星号的下标
    filler_index = NOTE_TO_INT()['*']

    # 处理语料库中的每一首编码歌曲
    for song in corpus:

        # 将星号添加到编码歌曲
        song = [filler_index]*segment_length + song + [filler_index]
        
        # 创建输入序列和对应的预测目标
        for i in range(len(song)-segment_length):
                
            segment = song[i: i+segment_length]
            target = song[i+segment_length]

            input_notes.append(segment)
            output_notes.append(target)
    
    # 将数字转化为独热向量
    input_notes = to_categorical(input_notes, num_classes=len(NOTE_TO_INT()))
    output_notes = to_categorical(output_notes, num_classes=len(NOTE_TO_INT()))

    return input_notes, output_notes


def build_model(weights_path=None):
    """构建RNN模型。

    :param weights_path (str): 模型权重的路径

    :return model: RNN模型
    """

    # 创建模型架构
    input_layer = Input(shape=(SEGMENT_LENGTH, len(NOTE_TO_INT())), 
                        name='input_layer')
    rnn_layer = LSTM(units=RNN_SIZE, 
                     name='rnn_layer')(input_layer)
    output_layer = Dense(units=len(NOTE_TO_INT()), 
                         activation="softmax", 
                         name='output_layer')(rnn_layer)

    model = Model(input_layer, output_layer)

    # 编译模型
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy')

    # 总结模型或者加载权重
    if weights_path==None:

        model.summary()

    else:

        model.load_weights(weights_path)

    return model


def train_model(input_notes, output_notes, weights_path=WEIGHTS_PATH):
    """训练并保存RNN模型。

    :param input_notes (ndarray): 训练集的输入序列集合
    :param output_notes (ndarray): 训练集的预测目标集合
    :param weights_path (str): 模型权重的路径

    :return:
    """

    # 构建模型
    model = build_model()

    # 尝试加载模型权重，读取失败则删除
    if os.path.exists(weights_path):
        
        try:

            model.load_weights(weights_path)
            print("模型权重加载成功")
        
        except:

            os.remove(weights_path)
            print("旧模型权重已删除")

    print("在%d个样本上训练" %(input_notes.shape[0]))

    # 保存模型权重
    checkpoint = ModelCheckpoint(filepath=weights_path,
                                 monitor='loss',
                                 verbose=0,
                                 save_best_only=True,
                                 mode='min')

    # 训练模型
    model.fit(x=input_notes,
              y=output_notes,
              batch_size=BATCH_SIZE,
              epochs=EPOCHS,
              callbacks=[checkpoint])


if __name__ == "__main__":

    input_notes, output_notes = create_training_data()
    train_model(input_notes, output_notes)