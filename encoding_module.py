import os
import pickle
from music21 import converter
from music21 import key
from music21 import interval
from music21 import pitch
from music21 import note
from music21 import chord
from utils import *

def transpose(score):
    """将音乐移调至C大调/a小调。

    :param score: 原本的乐谱

    :return score: 移调后的乐谱
    """

    for element in score.recurse():
        
        # 找到调号
        if isinstance(element, key.Key):

            # 找到主音
            if element.mode == 'major':
                
                tonic = element.tonic

            else:

                tonic = element.parallel.tonic

            # 根据主音进行移调
            gap = interval.Interval(tonic, pitch.Pitch('C'))
            score = score.transpose(gap)

            break
        
        # 未找到调号
        elif isinstance(element, note.Note) or \
             isinstance(element, note.Rest) or \
             isinstance(element, chord.Chord):
            
            break
        
        # 其他元素
        else:

            continue
    
    return score


def encode_data(dataset_path=DATASET_PATH):
    """编码数据集文件夹下的符号音乐文件。

    :param dataset_path (str): 数据集的路径

    :return data, filenames (list): 编码歌曲及其文件名
    """

    # 编码歌曲及其文件名
    data = []
    filenames = []

    # 遍历数据集路径下的所有文件
    for dirpath, dirlist, filelist in os.walk(dataset_path):
        
        # 处理每一个文件
        for this_file in filelist:

            # 确保后缀合法
            if os.path.splitext(this_file)[-1] not in EXTENSION:

                continue
        
            # 解析当前文件
            filename = os.path.join(dirpath, this_file)

            try:

                score = converter.parse(filename)

            except:

                print("警告：无法读取文件\"%s\"" %filename)
                continue
            
            print("正在解析\"%s\"" %filename)

            # 保留当前音乐中的第一个声部（通常是旋律部分）
            score = score.parts[0].flat

            # 移调至C大调/a小调
            score = transpose(score)

            # 编码歌曲序列
            song = []

            # 处理编码歌曲中的每一个音符（和弦）
            for element in score.recurse():
                
                if isinstance(element, note.Note):

                    note_pitch = element.pitch.midi
                    note_duration = element.quarterLength

                elif isinstance(element, note.Rest):

                    note_pitch = 0
                    note_duration = element.quarterLength
                    
                elif isinstance(element, chord.Chord):

                    note_pitch = element.notes[-1].pitch.midi
                    note_duration = element.quarterLength

                else:

                    continue
                
                # 确保时值合法
                if note_duration%0.25 == 0:

                    # 编码音符
                    note_step = int(note_duration/0.25)
                    song += [str(note_pitch)] + ['-']*(note_step-1)
                
                else:
                    
                    # 发现不可接受的时值
                    song = None
                    print("警告：在读取\"%s\"时发现不可接受的时值" %filename)
                    break
            
            if song!=None:

                 # 保存编码歌曲及其文件名
                data.append(song)
                filenames.append(os.path.splitext(os.path.basename(filename))[0])

    print("成功编码%d首歌曲" %(len(data)))

    return data, filenames


def save_corpus(data, corpus_path=CORPUS_PATH, vocabulary_path=VOCABULARY_PATH):
    """转换并保存语料库。

    :param data (list): 编码歌曲数据
    :param corpus_path (str): 语料库的路径
    :param vocabulary_path (str): 词表的路径

    :return:
    """

    # 创建并保存词表（附带星号“*”）
    vocabulary = sorted(set(sum(data, [])+['*']))
    with open(vocabulary_path, 'w') as filepath:
        json.dump(vocabulary, filepath, indent=4)

    # 将编码歌曲数据中的每一个音符映射成数字，并保存为语料库
    note2int = NOTE_TO_INT()
    corpus = [[note2int[note] for note in song] for song in data]
    with open(corpus_path, "wb") as filepath:
        pickle.dump(corpus, filepath)


if __name__ == "__main__":

    data, filenames = encode_data()
    save_corpus(data)