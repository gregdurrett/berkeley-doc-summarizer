#!/bin/bash

cd /Users/gdurrett/n/berkeley-doc-summarizer/rouge/ROUGE
config_file=$1
perl ROUGE-1.5.5.pl -e data/ -n 2 -x -m -s $config_file 1 | grep Average

