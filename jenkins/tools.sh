#!/bin/bash -f

exitError()
{
    test -e /tmp/tmp.${user}.$$
    if [ $? -eq 0 ] ; then
        rm -f /tmp/tmp.${user}.$$ 1>/dev/null 2>/dev/null
    fi
    echo "ERROR $1: $3" 1>&2
    echo "ERROR     LOCATION=$0" 1>&2
    echo "ERROR     LINE=$2" 1>&2
    exit $1
}
