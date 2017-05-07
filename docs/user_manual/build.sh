#!/bin/bash

if [ ! -d pandoc-bootstrap-adaptive-template ]; then
    git clone https://github.com/diversen/pandoc-bootstrap-adaptive-template
fi

if [ ! -d pandocfilters ]; then
    https://github.com/mbianco/pandocfilters.git
fi

#GT_doc_structure.md 
list_md_files="defines.md GT_doc_structure.md Installation.md Quick_Start_Guide.md accessor.md expandable_parameters.md conditional_switches.md"


## How to generate html with highgligh.js in order to highgligh GT keywords
PD_OPTIONS="--template pandoc-bootstrap-adaptive-template/standalone.html --css pandoc-bootstrap-adaptive-template/template.css --toc --toc-depth=2 --no-highlight"

pandoc -s highlight_js.md ${list_md_files} $PD_OPTIONS --from markdown --to json | python ./filters/note.py | runhaskell ./filters/IncludeFilter.hs| pandoc --from json --to html $PD_OPTIONS > index.html


## How to generate pdf
#pandoc --latex-engine=xelatex  -s ${list_md_files} -o index_md.pdf   --toc --toc-depth=2 --highlight-style pygments

## How to generate latex
#pandoc --to=latex  -s ${list_md_files} -o index_md.tex   --toc --toc-depth=2 --highlight-style pygments
