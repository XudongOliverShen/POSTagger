# part-of-speech (POS) tagger with char+word embedding and BiLSTM

## Introduction

This project implements a part-of-speech (POS) tagger. It uses a convolutional neural network for character-level embedding, along with word-level embedding. POS tagging is achieved by bidirectional LSTM. And it is super fast!

## Usage

Firstly clone this project from this repo.

To build the tagger

```python
python3.5 buildtagger.py sents.train tagger
```

To tag corpus

```python
python3.5 runtagger.py sents.test tagger sents.out
```

To evaluate results

```python
python3.5 eval.py sents.out sents.answer
```

## Contributors

Xudong Shen (xudong.shen@u.nus.edu)
