import json

# 路径设置
DATASET_PATH = "dataset"
CORPUS_PATH = "corpus.bin"
VOCABULARY_PATH = 'vocabulary.json'
WEIGHTS_PATH = 'weights.hdf5'
INPUTS_PATH = "inputs"
OUTPUTS_PATH = "outputs"

# 符号音乐编码模块的参数
EXTENSION = ['.musicxml', '.xml', '.mxl', '.midi', '.mid', '.krn']

# RNN模型的参数
SEGMENT_LENGTH = 128
RNN_SIZE = 256
BATCH_SIZE = 512
EPOCHS = 30

# MIDI生成器的参数
MAX_BARS = 0
TEMPERATURE = 0.8
MIDI_NUM = 3

def NOTE_TO_INT(vocabulary_path=VOCABULARY_PATH):
    """加载将音符映射为数字的词典。

    :param vocabulary_path (str): 词表的路径

    :return note2int (dict): 音符到数字的映射
    """

    with open(vocabulary_path, 'r') as filepath:
        vocabulary = json.load(filepath)
    
    return dict((note, index) for index, note in enumerate(vocabulary))


def INT_TO_NOTE(vocabulary_path=VOCABULARY_PATH):
    """加载将数字映射为音符的词典。
    
    :param vocabulary_path (str): 词表的路径

    :return int2note (dict): 数字到音符的映射
    """

    with open(vocabulary_path, 'r') as filepath:
        vocabulary = json.load(filepath)
    
    return dict((index, note) for index, note in enumerate(vocabulary))