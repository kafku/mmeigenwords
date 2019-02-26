#!/bin/bash
set -eu

# download files from onedrive, or you can acess directrly to
# https://1drv.ms/u/s!AnwqImTfVnOs22WmAZQF_Y4FGkkM
wget --no-check-certificate -O ./inputs.7z "https://onedrive.live.com/download?cid=AC7356DF64222A7C&resid=AC7356DF64222A7C%2111749&authkey=AAHuOht2wFTuE5o"
7z x ./inputs.7z
rm ./inputs.7z


# download benchmark datasets
## wordsim353
wget http://www.cs.technion.ac.il/~gabr/resources/data/wordsim353/wordsim353.zip
unzip wordsim353.zip
rm wordsim353.zip

## analogy task
wget https://raw.githubusercontent.com/tmikolov/word2vec/master/questions-words.txt
