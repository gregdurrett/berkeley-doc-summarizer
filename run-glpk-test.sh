#!/bin/bash

jarpath="berkeley-doc-summarizer-assembly-1.jar"
echo "Running with jar located at $jarpath"
# Extracts the existing java library path and adds /usr/local/lib/jni to it
java_lib_path=$(java -cp $jarpath edu.berkeley.nlp.summ.GLPKTest noglpk | head -1)
java_lib_path="$java_lib_path:/usr/local/lib/jni"
echo "Using the following library path: $java_lib_path"
java -ea -server -Xmx3g -Djava.library.path=$java_lib_path -cp $jarpath edu.berkeley.nlp.summ.GLPKTest

