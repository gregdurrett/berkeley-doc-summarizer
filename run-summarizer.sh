#!/bin/bash

jarpath="target/scala-2.11/berkeley-doc-summarizer-assembly-1.jar"
# Extracts the existing java library path
java_lib_path=$(java -cp $jarpath edu.berkeley.nlp.summ.GLPKTest noglpk | head -1)
java_lib_path="$java_lib_path:/usr/local/lib/jni"
echo "Using the following library path: $java_lib_path"

if [ -d "test-summaries-extractive" ]; then
  rm -rf test-summaries-extractive
fi
mkdir test-summaries-extractive/
# See edu.berkeley.nlp.summ.Summarizer for additional command line arguments
java -ea -server -Xmx3g -Djava.library.path=$java_lib_path -cp $jarpath edu.berkeley.nlp.summ.Summarizer -inputDir "test/" -outputDir "test-summaries-extractive/" -modelPath "models/summarizer-extractive.ser.gz" -noRst

if [ -d "test-summaries-extractive-compressive" ]; then
  rm -rf test-summaries-extractive-compressive
fi
mkdir test-summaries-extractive-compressive
java -ea -server -Xmx3g -Djava.library.path=$java_lib_path -cp $jarpath edu.berkeley.nlp.summ.Summarizer -inputDir "test/" -outputDir "test-summaries-extractive-compressive" -modelPath "models/summarizer-extractive-compressive.ser.gz"

if [ -d "test-summaries-full" ]; then
  rm -rf test-summaries-full
fi
mkdir test-summaries-full/
java -ea -server -Xmx3g -Djava.library.path=$java_lib_path -cp $jarpath edu.berkeley.nlp.summ.Summarizer -inputDir "test/" -outputDir "test-summaries-full"

