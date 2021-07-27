A simple LSTM-based music generation system.\
\
The system uses a dataset from [jukedeck/nottingham-dataset](https://github.com/jukedeck/nottingham-dataset), containing a total of 1034 songs, stored as a compressed file in the `dataset` folder, which should be extracted into the `dataset` folder when you need to use it.

## Install Dependencies
Python: 3.7.9\
Keras: 2.3.0\
tensorflow-gpu: 2.2.0\
music21: 6.7.1\
\
PS: Third party libraries can be installed using the `pip install` command.

## Generate Samples
This music generation system can be used to generate music in two ways: generate from scratch or based on existed music.

### Generate from Scratch
When there are no symbolic music files in the `inputs` folder, running `midi_generator.py` will generate multiple MIDI music tracks from zero into the `outputs` folder, the number generated depends on the parameter `MIDI_NUM`.

### Generate based on Exsited Music
When there are symbolic music files in the `inputs` folder, after running `midi_generator.py`, the system will continuation and generate MIDI music to the `outputs` folder based on this music, the new generated music contains all the notes of the original song (if the original song is not in C major/A minor, it will be transposed).

## Use Your Own Dataset
1.　Store all symbolic music files in the `dataset` folder, if you only have a small amount of data yourself, you can keep the original dataset of this system;\
2.　Run `encoding_module.py`, which will give you the vocabulary `vocabulary.json` of the dataset, and the corpus `corpus.bin`; \
3.　Run `rnn_model.py`, which will give you the model weights `weights.hdf5` after training (estimated time here is 2 hours).\
\
After that, you can use `midi_generator.py` to generate music that matches the musical style of the new dataset. \
\
If you need to finetune the parameters, you can do so in `utils.py`. It is not recommended to change the parameters in other files.
