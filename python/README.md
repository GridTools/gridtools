Gridtools4py

Python bindings/interpreter to the C++ Gridtools library


INSTALLATION

 ???

First install libconfig and its development files.

Redhat:     yum install libconfig libconfig-devel python-devel -y
Debian:     apt-get libconfig libconfig-devel python-devel
openSUSE:   zypper install libconfig8 libconfig-devel python-devel
            (available in the Packman repository)

On other platforms, you can compile libconfig from source:

    http://www.hyperrealm.com/libconfig/

Instructions:

    wget http://www.hyperrealm.com/libconfig/libconfig-1.3.2.tar.gz
    tar -zxf libconfig-1.3.2.tar.gz
    cd libconfig-1.3.2
    export MYPREFIX=/usr
    # or if you lack privileges for making system-wide changes:
    # export MYPREFIX=$HOME/local
    ./configure --prefix=$MYPREFIX
    make
    make install

To install this module, run the following commands:

    python setup.py build
    python setup.py install
    

SEE ALSO

You can also look for Perl bindings information at:

    RT, CPAN's request tracker
        http://rt.cpan.org/NoAuth/Bugs.html?Dist=Conf-Libconfig

    AnnoCPAN, Annotated CPAN documentation
        http://annocpan.org/dist/Conf-Libconfig

    CPAN Ratings
        http://cpanratings.perl.org/d/Conf-Libconfig

    Search CPAN
        http://search.cpan.org/dist/Conf-Libconfig/


COPYRIGHT AND LICENCE

Copyright (C) 2014 - CSCS

This program is released under the ??? license
