#!/bin/bash

if [ "`hostname | grep greina`" != "" ] ; then
    myhost="greina"
elif [ "`hostname | grep kesch`" != "" ] ; then
    myhost="kesch"
elif [ "`hostname | grep dom`" != "" ] ; then
    myhost="dom"
else
    echo "ERROR: host not known"
    exit 1
fi
