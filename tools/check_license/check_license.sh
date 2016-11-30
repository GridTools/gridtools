#!/bin/bash
# run this script from the root directory
if [ ! -f $PWD/LICENSE ] ; then
    echo "you have to run the scripto ./tools/check_license/check_license.sh from the root of the sources"
    exit -1
fi

for file in `find . -regextype posix-egrep -regex ".*\.(hpp|cpp|cu)$" `; do
	if ! grep -q -f $PWD/LICENSE $file ; then
            echo $file
        fi
done
