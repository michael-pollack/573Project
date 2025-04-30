#!/bin/bash



pip install textstat
pip install numpy==1.24.1
pip install pandas==1.5.3
pip install protobuf==6.30.2
pip install lens-metric
pip install rouge-score
pip install bert-score
pip install spacy==3.7.5
pip install radgraph
pip install f1chexbert
pip install evaluate
pip install sentence-transformers==4.0.2

python -m spacy download en

#alignscore
git clone https://github.com/yuh-zha/AlignScore.git
#summac
git clone https://github.com/tingofurro/summac.git

