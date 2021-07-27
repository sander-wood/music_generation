import json

# paths
DATASET_PATH = "dataset"
CORPUS_PATH = "corpus.bin"
VOCABULARY_PATH = 'vocabulary.json'
WEIGHTS_PATH = 'weights.hdf5'
INPUTS_PATH = "inputs"
OUTPUTS_PATH = "outputs"

# parameters for encoding_module.py
EXTENSION = ['.musicxml', '.xml', '.mxl', '.midi', '.mid', '.krn']

# parameters for rnn_model.py
SEGMENT_LENGTH = 128
RNN_SIZE = 256
BATCH_SIZE = 512
EPOCHS = 30

# parameters for midi_generator.py
MAX_BARS = 0
TEMPERATURE = 0.8
MIDI_NUM = 3

def NOTE_TO_INT(vocabulary_path=VOCABULARY_PATH):
    """Loads note2int dictionary.

    :param vocabulary_path (str): Path to vocabulary

    :return note2int (dict): Note-to-int mapping
    """

    with open(vocabulary_path, 'r') as filepath:
        vocabulary = json.load(filepath)
    
    return dict((note, index) for index, note in enumerate(vocabulary))


def INT_TO_NOTE(vocabulary_path=VOCABULARY_PATH):
    """Loads int2note dictionary.
    
    :param vocabulary_path (str): Path to vocabulary

    :return int2note (dict): Int-to-note mapping
    """

    with open(vocabulary_path, 'r') as filepath:
        vocabulary = json.load(filepath)
    
    return dict((index, note) for index, note in enumerate(vocabulary))