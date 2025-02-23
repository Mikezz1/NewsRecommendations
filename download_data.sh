#!/bin/bash

mkdir data
cd data

# Dowload GloVe pre-trained word embedding and unzip
wget https://huggingface.co/stanfordnlp/glove/resolve/main/glove.840B.300d.zip
unzip glove.840B.300d.zip

# Download MIND-small dataset and unzip
wget https://mind201910small.blob.core.windows.net/release/MINDsmall_train.zip
wget https://mind201910small.blob.core.windows.net/release/MINDsmall_dev.zip
unzip MINDsmall_train.zip -d MINDsmall_train
unzip MINDsmall_dev.zip -d MINDsmall_dev

rm glove.840B.300d.zip
rm MINDsmall_train.zip
rm MINDsmall_dev.zip


wget https://www.dropbox.com/s/z839s0kgjnm5ekl/news_pop.csv?dl=0

echo 'Data download finish.'