#!/bin/bash

#
# remove any files left from the previous run
#
rm -rf /tmp/__gridtools_* 

#
# run the tests, including a coverage report
#
nosetests -v -s -x --with-coverage --cover-package=gridtools --cover-erase --cover-html tests.test_combined_stencils tests.test_stencils tests.test_sw
