import datetime
import numpy as np
from music21 import stream
from rnn_model import *

note2int = NOTE_TO_INT()
int2note = INT_TO_NOTE()

def from_scratch(midi_num=MIDI_NUM):
    """为自动生成创建空数据。

    :param midi_num (int): 需要生成的MIDI文件的数量
    
    :return data (list): 空的初始输入序列
    :return filenames (list): 根据当前时间生成的文件名
    """

    # 编码歌曲及其文件名
    data = []
    filenames = []

    # 读取当前时间
    nowTime = datetime.datetime.utcnow().strftime('%Y-%m-%d-%H-%M-%S')

    # 创建带有文件名的空数据
    for midi_index in range(midi_num):

        data.append([])
        filenames.append(nowTime+'-'+str(midi_index+1))
    
    return data, filenames


def sample(prediction, temperature=TEMPERATURE):
    """对RNN模型输出的概率向量进行采样。

    :param prediction (ndarray): RNN模型输出的概率向量
    :param temperature (float): 等于0时相当于贪婪采样，而等于无穷时相当于均匀采样
    
    :return index (int): 采样结果的下标
    """

    # 改变概率的分布
    prediction = np.log(prediction) / temperature
    probabilites = np.exp(prediction) / np.sum(np.exp(prediction))

    # 随机采样
    index = np.random.choice(range(len(probabilites)), p=probabilites)

    return index


def generate_notes(model, data, filenames, max_notes=MAX_BARS*16):
    """使用RNN模型生成音符。

    :param model: RNN模型
    :param data (list): 编码歌曲数据
    :param filenames (list): 编码歌曲的文件名
    :param max_notes (int): 生成音符数量的上限
    
    :return:
    """

    # 处理数据中的每一首编码歌曲
    for index in range(len(data)):
        
        # 将星号添加到编码歌曲
        song = ['*']*SEGMENT_LENGTH + data[index]

        # 将编码歌曲数据中的每一个音符映射成数字
        try:

            input_notes = [note2int[note] for note in song[-SEGMENT_LENGTH:]]

        except:

            print("警告：在\"%s\"中发现未知音符" %filenames[index])
            continue
        
        midi_path = OUTPUTS_PATH+ '\\' + filenames[index] + '.mid'
        print("正在处理\"%s\"" %midi_path)

        output_note = None
        num_notes = 0

        # 若未采样到星号“*”或者未超过最大音符数量上限，则继续进行生成
        while (num_notes<max_notes or max_notes==0) and output_note!='*':
            
            # 将输入序列独热向量化并添加一个维度
            one_hot_input = to_categorical(input_notes, num_classes=len(NOTE_TO_INT()))
            one_hot_input = one_hot_input[np.newaxis, ...]

            # 预测下一个音符
            prediction = model.predict(one_hot_input)[0]
            note_index = sample(prediction)
            output_note = int2note[note_index]

            # 更新输入序列
            input_notes = input_notes[1:]
            input_notes.append(note_index)

            num_notes += 1
            song.append(output_note)
        
        convert_midi(song[SEGMENT_LENGTH:], midi_path)


def convert_midi(song, midi_path):
    """转换编码歌曲为MIDI文件

    :param song (list): 编码歌曲
    :param midi_path (str): MIDI文件的路径
    
    :return:
    """

    # 初始化
    midi_notes = []
    pre_element = None
    duration = 0.0
    offset = 0.0
    
    # 解码编码歌曲
    for element in song:
        
        if element!='-':

            # 创建新音符
            if pre_element!=None:

                if pre_element=='0':

                    new_note = note.Rest()
                
                else:

                    new_note = note.Note(int(pre_element))
                
                new_note.quarterLength = duration
                new_note.offset = offset
                midi_notes.append(new_note)
            
            # 更新偏移量并保存当前音符
            offset += duration
            pre_element = element
            duration = 0.25
        
        else:
            
            # 更新时值
            duration += 0.25

    # 保存为MIDI文件
    score_stream = stream.Stream(midi_notes)
    score_stream.write('mid', fp=midi_path)
        

if __name__ == "__main__":

    # 进行自动生成或根据已有音乐文件执行续写生成
    if os.listdir(INPUTS_PATH):

        data, filenames = encode_data(INPUTS_PATH)
    
    else:

        data, filenames = from_scratch()

    model = build_model(WEIGHTS_PATH)
    generate_notes(model, data, filenames)