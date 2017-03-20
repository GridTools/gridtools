#!/bin/bash
# run this script from the root directory
if [ ! -f $PWD/LICENSE ] ; then
    echo "you have to run the script ./tools/check_license/check_license.sh from the root of the sources"
    exit -1
fi

for file in `find . -regextype posix-egrep -regex ".*\.(hpp|cpp|cu)$" `; do
    if ! grep -q "Copyright (c) 2016, GridTools Consortium" $file ; then
        echo $file
        if [ "$1" == "fix" ]; then
            echo "/*
$(cat LICENSE)
*/
$(cat $file)" > $file
        fi
        else # if the license matches the string above
        if [ "$1" == "update" ]; then
            sed -i '0,/\*\//d' $file
            echo "/*
$(cat LICENSE)
*/
$(cat $file)" > $file
        fi
    fi
done
