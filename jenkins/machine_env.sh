#!/bin/bash

if [ "`hostname | grep greina`" != "" ] ; then
    myhost="greina"
elif [ "`hostname | grep kesch`" != "" ] ; then
    myhost="kesch"
elif [ "`hostname | grep dom`" != "" ] ; then
    myhost="dom"
elif [ "`hostname | grep daint`" != "" ] ; then
    myhost="daint"
elif [ "`hostname | grep nid0`" != "" ] ; then
    myhost="daint"
else
    echo "ERROR: host not known"
    exit 1
fi
