#!/bin/bash

if [ -f "build_$1.sh" ]
then
    echo \"/build_$1.sh\"
    eval "./build_$1.sh"
fi