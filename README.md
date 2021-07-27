# 音乐生成系统
这个资源库包含了《音乐生成入门》系列视频中的音乐生成系统的代码。\
\
系统使用的数据集来自[jukedeck/nottingham-dataset](https://github.com/jukedeck/nottingham-dataset)，总共包含1034首音乐，以压缩文件的形式存放在`dataset`文件夹下，需要使用时请把它解压到`dataset`文件夹内。

## 环境配置
Python: 3.7.9\
Keras: 2.3.0\
tensorflow-gpu: 2.2.0\
music21: 6.7.1\
\
PS: 第三方库可以使用`pip install`命令安装

## 借助模型生成音乐
这个音乐生成系统可以采取自动生成和续写生成两种方式生成音乐。

### 自动生成
当`inputs`文件夹下没有任何符号音乐文件时，运行`midi_generator.py`后，系统会从零开始生成多首MIDI音乐至`outputs`文件夹下，生成的数量取决于参数`MIDI_NUM`。

### 续写生成
当`inputs`文件夹下存在符号音乐文件时，运行`midi_generator.py`后，系统会基于这些音乐进行续写并生成MIDI音乐至`outputs`文件夹下，新生成的音乐包含原曲的全部音符（若原曲不在C大调/a小调上，则会进行移调）。

## 替换自己的数据集
1.　将所有符号音乐文件存入`dataset`文件夹下，如果自己的数据量较小，可以保留这个系统原本的数据集；\
2.　运行`encoding_module.py`，将会得到词表`vocabulary.json`和语料库`corpus.bin`；\
3.　运行`rnn_model.py`，训练结束后会得到模型权重`weights.hdf5`（此处预计耗时2小时）;\
\
之后就可以使用`midi_generator.py`生成出符合新数据集音乐风格的作品了。\
\
如果需要调整参数可以在`utils.py`中进行修改，不推荐在其他文件下修改参数。
