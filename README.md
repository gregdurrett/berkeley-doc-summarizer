berkeley-doc-summarizer
=======================

The Berkeley Document Summarizer is a learning-based single-document
summarization system.  It compresses source document text based on constraints
from constituency parses and RST discourse parses. Moreover, it can improve
summary clarity by reexpressing pronouns whose antecedents would otherwise be
deleted or unclear.



##Preamble

The Berkeley Document Summarizer is described in:

"Learning-Based Single-Document Summarization with Compression and Anaphoricity Constraints"
Greg Durrett, Taylor Berg-Kirkpatrick, and Dan Klein. ACL 2016.

See http://www.eecs.berkeley.edu/~gdurrett/ for papers and BibTeX.

Questions? Bugs? Email me at gdurrett@eecs.berkeley.edu



##License

Copyright (c) 2013-2016 Greg Durrett. All Rights Reserved.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see http://www.gnu.org/licenses/


##Setup

####Models and Data

Models are not included in GitHub due to their large size. Download the latest
models from http://nlp.cs.berkeley.edu/projects/summarizer.shtml. These
are necessary for both training the system (you need the EDU segmenter, discourse
parser, and coreference model) as well as running it (you need the EDU segmenter,
discourse parser, and summarization model, which contains the coreference model).
All of these are expected in the models/ subdirectory.

We also require [number and gender data](http://www.clsp.jhu.edu/~sbergsma/Gender/)
(default path the system expects this data at: data/gender.data).

####GLPK

For solving ILPs, our system relies on GLPK.  The easiest way to install GLPK
is with [homebrew](http://brew.sh/). Additionally, when running the system, you
must have ```glpk-java-1.1.0.jar``` on the build path (already done if you're
using sbt) as well as make sure the Java Native Interface (JNI) libraries are
accessible. These allow the GLPK Java bindings to interact with the native GLPK
code.

You can try this out with ```edu.berkeley.nlp.summ.GLPKTest```; if this class runs
without error, you're good! If you do get an error, you may need to augment the
Java library path with the location of the libglpk_java libraries as follows:

    -Djava.library.path="<current library path>:<location of libglpk_java libraries>"
    
On OS X, this may be located in ```/usr/local/lib/jni```. ```run-summarize.sh```
attempts to set this automatically, but may not work for your system.

####Building from source

The easiest way to build is with SBT:
https://github.com/harrah/xsbt/wiki/Getting-Started-Setup

then run

    sbt assembly

which will compile everything and build a runnable jar.

You can also import it into Eclipse and use the Scala IDE plug-in for Eclipse
http://scala-ide.org



##Running the system

The two most useful main classes are ```edu.berkeley.nlp.summ.Main``` and
```edu.berkeley.nlp.summ.Summarizer```. The former is a more involved harness
for training and evaluating the system on the New York Times corpus (see below
for how to acquire this corpus), and the latter simply takes a trained model
and runs it. Both files contain descriptions of their functionality and command-line
arguments.

An example run on new data is included in ```run-summarizer.sh```. The main
prerequisite for running the summarizer on new data is having that data preprocessed
in the CoNLL format with constituency parses, NER, and coreference. For a system that
does this, see the [Berkeley Entity Resolution System](https://github.com/gregdurrett/berkeley-entity).
The ```test/``` directory already contains a few such files.

The system is distributed with several pre-trained variants:

* summarizer-extractive.ser.gz: a sentence-extractive summarizer
* summarizer-extractive-compressive.ser.gz: an extractive-compressive summarizer
* summarizer-full.ser.gz: an extractive-compressive summarizer with the ability to rewrite pronouns
and additional coreference features and constraints



##Training

####New York Times Dataset

The primary corpus we use for training and evaluation is the New York Times Annotated Corpus
(Sandhaus, 2007), LDC2008T19. We distribute our preprocessing as standoff annotations which
replace words with (line, char start, char end) triples, except for some cases where words are
included manually (e.g. when tokenization makes our data non-recoverable from the original
file). A few scattered tokens are included explicitly, plus roughly 1% of files that our
system couldn't find a suitable alignment for.

To prepare the dataset, first you need to extract all the XML files from 2005-2007 and flatten
them into a single directory. Not all files have summaries, so not all of these will
be used. Next, run

    java -Xmx3g -cp <jarpath> edu.berkeley.nlp.preprocess.summ.preprocess.StandoffAnnotationHandler
      -inputDir <path_to_standoffs> -rawXMLPath <path_to_flattened_NYT_XMLs> -outputDir <output>

This will take a standoff annotation directory (either train or test) and reconstitute
the files using the XML data, writing to the output directory.
    
To reconstitute abstracts, run:

    java -Xmx3g -cp <jarpath> edu.berkeley.nlp.preprocess.summ.preprocess.StandoffAnnotationHandler
      -inputDir <path_to_standoff_abstracts> -rawXMLPath <path_to_flattened_NYT_XMLs> -outputDir <output>
      -tagName "abstract"

####ROUGE Scorer

We bundle the system with a version of the ROUGE scorer that will be called during
execution. ```rouge-gillick.sh``` hardcodes command-line arguments used in this work and
in Hirao et al. (2013)'s work.

See ```edu.berkeley.nlp.summ.RougeComputer.evaluateRougeNonTok``` for a method you can
use to evaluate ROUGE in a manner consistent with our evaluation.

####Training the system

To train the full system, run:

    java -Xmx80g -cp <jarpath> -Djava.library.path=<library path>:/usr/local/lib/jni edu.berkeley.nlp.summ.Main
      -trainDocsPath <path_to_train_conll_docs> -trainAbstractsPath <path_to_train_summaries>
      -evalDocsPath <path_to_eval_conll_docs> -evalAbstractsPath <path_to_eval_summaries> -abstractsAreConll
      -modelPath "models/trained-model.ser.gz" -corefModelPath "models/coref-onto.ser.gz"
      -printSummaries -printSummariesForTurk

where <jarpath>, <library path>, and the data paths are instantiated accordingly. The system requires a lot
of memory due to caching 25,000 training documents with annotations.

To train the sentence extractive version of the system, add:

    -doPronounReplacement false -useFragilePronouns false -noRst

To train the extractive-compressive version, add:

    -doPronounReplacement false -useFragilePronouns false


The results you get using this command should be:

* extractive: ROUGE-1 recall: 38.6 / ROUGE-2 recall: 23.3
* extractive-compressive: XXX
* full: XXX

(Results are slightly different from those in the paper due to minor changes for this
release.)

