#!/bin/bash

if [ ! -d pandoc-bootstrap-adaptive-template ]; then
    git clone https://github.com/diversen/pandoc-bootstrap-adaptive-template
fi

list_md_files="defines.md Installation.md Quick_Start_Guide.md GT_doc_structure.md accessor.md"

## How to generate html
#pandoc -s ${list_md_files} -o index.html --template pandoc-bootstrap-adaptive-template/standalone.html --css pandoc-bootstrap-adaptive-template/template.css --toc --toc-depth=2 --highlight-style pygments

## How to generate html with highgligh.js in order to highgligh GT keywords
pandoc -s highlight_js.md ${list_md_files} -o index.html --template pandoc-bootstrap-adaptive-template/standalone.html --css pandoc-bootstrap-adaptive-template/template.css --toc --toc-depth=2 --no-highlight

## How to generate pdf
#pandoc --latex-engine=xelatex  -s ${list_md_files} -o index_md.pdf   --toc --toc-depth=2 --highlight-style pygments

## How to generate latex
#pandoc --to=latex  -s ${list_md_files} -o index_md.tex   --toc --toc-depth=2 --highlight-style pygments
