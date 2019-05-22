#!/bin/bash
for file in `find . -regextype posix-egrep -regex ".*\.(hpp|cpp|cu)$" `; do 
	clang-format-6 $file > $file.tmp; 
	mv $file.tmp $file; 
done

