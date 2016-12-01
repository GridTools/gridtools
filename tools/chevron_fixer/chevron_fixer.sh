#!/bin/bash
DIR=/code/gridtools2/tools/chevron_fixer/
for file in `find . -regextype posix-egrep -regex ".*\.(hpp|cpp|cu)$" `; do 
	cat $file | ${DIR}/chevron_fixer_file.sh > $file.tmp
#gawk '{ if(match($0,/<[ \t]*<[ \t]*<.*>[ \t]*>[ \t]*>/)) { sub(/<[ \t]*<[ \t]*</,"<<<", $0); sub(/>[ \t]*>[ \t]*>/, ">>>", $0); } print $0 }' > $file.tmp; 
	mv $file.tmp $file; 
done

