#!/bin/bash

if [ ! -d ../pandoc_tools/pandoc-bootstrap-adaptive-template ]; then
    git clone https://github.com/diversen/pandoc-bootstrap-adaptive-template ../pandoc_tools/pandoc-bootstrap-adaptive-template
fi


if [ ! -d ../pandoc_tools/pandocfilters ]; then
    git clone https://github.com/mbianco/pandocfilters.git ../pandoc_tools/pandocfilters
fi

if ! [[ $PYTHONPATH =~ .*pandocfilters.* ]]; then
    echo pandocfilters do not seem to be in the path: inlcuding `pwd`/../pandoc_tools/pandocfilter folder
    export PYTHONPATH=`pwd`/../pandoc_tools/pandocfilters:$PYTHONPATH
fi

#GT_doc_structure.md 
list_md_files="../pandoc_tools/defines.md GT_doc_structure.md"


## How to generate html with highgligh.js in order to highgligh GT keywords
PD_OPTIONS="--template ../pandoc_tools/pandoc-bootstrap-adaptive-template/standalone.html --css ../pandoc_tools/pandoc-bootstrap-adaptive-template/template.css --toc --toc-depth=2 --no-highlight"

#pandoc -s highlight_js.md ${list_md_files} $PD_OPTIONS --from markdown --to json | python ./filters/note.py | runhaskell -v ./filters/IncludeFilter.hs| pandoc --from json --to html $PD_OPTIONS > index.html
pandoc -s ../pandoc_tools/highlight_js.md ${list_md_files} $PD_OPTIONS --from markdown --to json | python ../pandoc_tools/filters/include.py | python ../pandoc_tools/filters/note.py | pandoc --from json --to html $PD_OPTIONS > index.html


## How to generate pdf
#pandoc --latex-engine=xelatex  -s ${list_md_files} -o index_md.pdf   --toc --toc-depth=2 --highlight-style pygments

## How to generate latex
#pandoc --to=latex  -s ${list_md_files} -o index_md.tex   --toc --toc-depth=2 --highlight-style pygments
