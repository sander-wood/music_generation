import datetime
import numpy as np
from music21 import stream
from rnn_model import *

note2int = NOTE_TO_INT()
int2note = INT_TO_NOTE()

def from_scratch(midi_num=MIDI_NUM):
    """Creates empty encoded songs for generating music from scratch.

    :param midi_num (int): Number of MIDI files
    
    :return data (list): Empty initial music sequences
    :return filenames (list): Filenames generated based on the date time
    """

    # encoded songs and their file names
    data = []
    filenames = []

    # read current time
    nowTime = datetime.datetime.utcnow().strftime('%Y-%m-%d-%H-%M-%S')

    # create empty data with names
    for midi_index in range(midi_num):

        data.append([])
        filenames.append(nowTime+'-'+str(midi_index+1))
    
    return data, filenames


def sample(prediction, temperature=TEMPERATURE):
    """Sampling the probability vector output by RNN model.

    :param prediction (ndarray): Probability vector output by RNN model
    :param temperature (float): 0 is equivalent to argmax/max and inf is equivalent to uniform sampling
    
    :return index (int): Index of the sampled result
    """

    # change the distribution of probability
    prediction = np.log(prediction) / temperature
    probabilites = np.exp(prediction) / np.sum(np.exp(prediction))

    # random sampling
    index = np.random.choice(range(len(probabilites)), p=probabilites)

    return index


def generate_notes(model, data, filenames, max_notes=MAX_BARS*16):
    """Generates notes with RNN model.

    :param model: RNN model
    :param data (list): Encoded songs
    :param filenames (list): File names of the encoded songs
    :param max_notes (int): Maximum number of notes to be generated
    
    :return:
    """

    # process each song in data
    for index in range(len(data)):
        
        # append fillers to the song 
        song = ['*']*SEGMENT_LENGTH + data[index]

        # map each element in the song to int
        try:

            input_notes = [note2int[note] for note in song[-SEGMENT_LENGTH:]]

        except:

            print("Warning: Unknown notes found in \"%s\"" %filenames[index])
            continue
        
        midi_path = OUTPUTS_PATH+ '\\' + filenames[index] + '.mid'
        print("Processing \"%s\"" %midi_path)

        output_note = None
        num_notes = 0

        # generate note if no '*' is sampled or the number of notes does not exceed the limit
        while (num_notes<max_notes or max_notes==0) and output_note!='*':
            
            # one-hot vectorize input and add an axis to it
            one_hot_input = to_categorical(input_notes, num_classes=len(NOTE_TO_INT()))
            one_hot_input = one_hot_input[np.newaxis, ...]

            # predict the next note
            prediction = model.predict(one_hot_input)[0]
            note_index = sample(prediction)
            output_note = int2note[note_index]

            # update the input sequence
            input_notes = input_notes[1:]
            input_notes.append(note_index)

            num_notes += 1
            song.append(output_note)
        
        convert_midi(song[SEGMENT_LENGTH:], midi_path)


def convert_midi(song, midi_path):
    """Converts the encoded song to a midi file.

    :param song (list): Encoded song
    :param midi_path (str): Path to the midi file
    
    :return:
    """

    # initialization
    midi_notes = []
    pre_element = None
    duration = 0.0
    offset = 0.0
    
    # decode the song
    for element in song:
        
        if element!='-':

            # create new note
            if pre_element!=None:

                if pre_element=='0':

                    new_note = note.Rest()
                
                else:

                    new_note = note.Note(int(pre_element))
                
                new_note.quarterLength = duration
                new_note.offset = offset
                midi_notes.append(new_note)
            
            # update offset and save current note
            offset += duration
            pre_element = element
            duration = 0.25
        
        else:
            
            # update duration
            duration += 0.25

    # save as midi
    score_stream = stream.Stream(midi_notes)
    score_stream.write('mid', fp=midi_path)
        

if __name__ == "__main__":

    # generate music from scratch or based on existing file
    if os.listdir(INPUTS_PATH):

        data, filenames = encode_data(INPUTS_PATH)
    
    else:

        data, filenames = from_scratch()

    model = build_model(WEIGHTS_PATH)
    generate_notes(model, data, filenames)