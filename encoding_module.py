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
    """Transpose the score to C major/A minor.

    :param score: Original score

    :return score: Transposed score
    """

    for element in score.recurse():
        
        # key signature found
        if isinstance(element, key.Key):

            # tonic found
            if element.mode == 'major':
                
                tonic = element.tonic

            else:

                tonic = element.parallel.tonic

            # transpose score
            gap = interval.Interval(tonic, pitch.Pitch('C'))
            score = score.transpose(gap)

            break
        
        # no key signatrue found
        elif isinstance(element, note.Note) or \
             isinstance(element, note.Rest) or \
             isinstance(element, chord.Chord):
            
            break
        
        # other element
        else:

            continue
    
    return score


def encode_data(dataset_path=DATASET_PATH):
    """Encodes the symbloc music in the dataset folder.

    :param dataset_path (str): Path to the dataset

    :return data, filenames (list): Encoded songs and their file names
    """

    # encoded songs and their file names
    data = []
    filenames = []

    # loop through the dataset folder
    for dirpath, dirlist, filelist in os.walk(dataset_path):
        
        # process each file
        for this_file in filelist:

            # ensure extension is valid
            if os.path.splitext(this_file)[-1] not in EXTENSION:

                continue
        
            # parse the file
            filename = os.path.join(dirpath, this_file)

            try:

                score = converter.parse(filename)

            except:

                print("Warning: Failed to read \"%s\"" %filename)
                continue
            
            print("Parsing \"%s\"" %filename)

            # keep the first part (usually is the melody) of score
            score = score.parts[0].flat

            # transpose to C major/A minor
            score = transpose(score)

            # encoded song
            song = []

            # process each note (chord) in the score 
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
                
                # ensure duration is valid
                if note_duration%0.25 == 0:

                    # encode note
                    note_step = int(note_duration/0.25)
                    song += [str(note_pitch)] + ['-']*(note_step-1)
                
                else:
                    
                    # unacceptable duration found
                    song = None
                    print("Warning: Found an unacceptable duration when reading the file \"%s\"" %filename)
                    break
            
            if song!=None:

                 # save the encoded song and its name
                data.append(song)
                filenames.append(os.path.splitext(os.path.basename(filename))[0])

    print("Successfully encoded %d songs" %(len(data)))

    return data, filenames


def save_corpus(data, corpus_path=CORPUS_PATH, vocabulary_path=VOCABULARY_PATH):
    """Converts and saves the corpus.

    :param data (list): Encoded songs
    :param corpus_path (str): Path to the corpus
    :param vocabulary_path (str): Path to the vocabulary

    :return:
    """

    # create and save vocabulary (with the filler '*')
    vocabulary = sorted(set(sum(data, [])+['*']))
    with open(vocabulary_path, 'w') as filepath:
        json.dump(vocabulary, filepath, indent=4)

    # map each element in data to int and save as corpus
    note2int = NOTE_TO_INT()
    corpus = [[note2int[note] for note in song] for song in data]
    with open(corpus_path, "wb") as filepath:
        pickle.dump(corpus, filepath)


if __name__ == "__main__":

    data, filenames = encode_data()
    save_corpus(data)