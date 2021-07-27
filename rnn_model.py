from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Dense
from keras import Model
from keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.utils.np_utils import to_categorical
from encoding_module import *

def create_training_data(segment_length=SEGMENT_LENGTH, corpus_path=CORPUS_PATH):
    """Creates training data for the RNN model.

    :param segment_length (int): Length of each segment we want to divide the encoded data into
    :param corpus_path (str): Path to the corpus

    :return input_notes (ndarray): Input part of training data
    :return output_notes (ndarray): Output part of training data
    """

    # load corpus
    with open(corpus_path, "rb") as filepath:
        corpus = pickle.load(filepath)

    # input and output of training data
    input_notes = []
    output_notes = []

    # get the index of filler
    filler_index = NOTE_TO_INT()['*']

    # process each song in the corpus
    for song in corpus:

        # append fillers to the song 
        song = [filler_index]*segment_length + song + [filler_index]
        
        # create segment-target pairs
        for i in range(len(song)-segment_length):
                
            segment = song[i: i+segment_length]
            target = song[i+segment_length]

            input_notes.append(segment)
            output_notes.append(target)
    
    # one-hot vectorize input and output
    input_notes = to_categorical(input_notes, num_classes=len(NOTE_TO_INT()))
    output_notes = to_categorical(output_notes, num_classes=len(NOTE_TO_INT()))

    return input_notes, output_notes


def build_model(weights_path=None):
    """Builds RNN model.

    :param weights_path (str): Path to weights of RNN model

    :return model: RNN model
    """

    # create model architecture
    input_layer = Input(shape=(SEGMENT_LENGTH, len(NOTE_TO_INT())), 
                        name='input_layer')
    rnn_layer = LSTM(units=RNN_SIZE, 
                     name='rnn_layer')(input_layer)
    output_layer = Dense(units=len(NOTE_TO_INT()), 
                         activation="softmax", 
                         name='output_layer')(rnn_layer)

    model = Model(input_layer, output_layer)

    # compile model
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy')

    # summarise model or load weights
    if weights_path==None:

        model.summary()

    else:

        model.load_weights(weights_path)

    return model


def train_model(input_notes, output_notes, weights_path=WEIGHTS_PATH):
    """Trains and saves RNN model.

    :param input_notes (ndarray): Input part of training data
    :param output_notes (ndarray): Output part of training data
    :param weights_path (str): Path to weights of RNN model

    :return:
    """

    # build RNN model
    model = build_model()

    # load weights or delete it
    if os.path.exists(weights_path):
        
        try:

            model.load_weights(weights_path)
            print("checkpoint loaded")
        
        except:

            os.remove(weights_path)
            print("checkpoint deleted")

    print("Train on %d samples" %(input_notes.shape[0]))

    # save weights
    checkpoint = ModelCheckpoint(filepath=weights_path,
                                 monitor='loss',
                                 verbose=0,
                                 save_best_only=True,
                                 mode='min')

    # train model
    model.fit(x=input_notes,
              y=output_notes,
              batch_size=BATCH_SIZE,
              epochs=EPOCHS,
              callbacks=[checkpoint])


if __name__ == "__main__":

    input_notes, output_notes = create_training_data()
    train_model(input_notes, output_notes)