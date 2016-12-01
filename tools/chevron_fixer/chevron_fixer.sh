#!/bin/bash
if [ -z ${BASH} ]; then
  echo "ERROR: need to execute the script in bash" 
  exit 1
fi
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
for file in `find . -regextype posix-egrep -regex ".*\.(hpp|cpp|cu)$" `; do 
	cat $file | ${DIR}/chevron_fixer_file.sh > $file.tmp
	mv $file.tmp $file; 
done

