# -*- coding: utf-8 -*-
import logging

import numpy as np



class Utilities ( ):
    """
    Class to contain various helpful functions.
    Currently contains floating point precision validation.-
    """
    def __init__ (self):
        self.name = self.__class__.__name__.capitalize ( )
        #
        # these entities are automatically generated at compile time
        #
        self.ulib_file  = None
        self.ucpp_file  = None
        self.umake_file = None
        #
        # a reference to the compiled dynamic library
        #
        self.ulib_obj = None


    def validate_translate (self):
        """
        Creates the C++ source file containing utilities to be used by Python.
        Currently only consists of the function for checking the size of the floating
        point type used in the backend.
        """
        from gridtools import JinjaEnv

        #
        # instantiate each of the templates and render them
        #
        cpp  = JinjaEnv.get_template ("utilities.cpp")
        make = JinjaEnv.get_template ("Makefile.utilities")

        namespace = self.name.lower ( )

        return (cpp.render  (stencil=self),
                make.render (stencil=self))


    def generate_validation_code (self):
        """
        Generates native code for validating the precision of the floating
        point type.
            src_dir     directory where the files should be saved (optional).-
        """
        from os        import write, path, makedirs
        from tempfile  import mkdtemp
        from gridtools import JinjaEnv
        #
        # create directory and files for the generated code
        #
        self.src_dir = mkdtemp (prefix="__gridtools_")

        namespace       = self.name.lower ( )
        self.ulib_file  = 'libutilities.so'
        self.ucpp_file  = 'utilities.cpp'
        self.umake_file = 'Makefile.utilities'

        logging.info ("Generating backend float type check code (C++) in '%s'" % self.src_dir)

        #
        # code for the stencil, the library entry point and makefile
        #
        cpp_src, make_src = self.validate_translate ( )
        with open (path.join (self.src_dir, self.ucpp_file), 'w') as cpp_hdl:
            cpp_hdl.write (cpp_src)
        with open (path.join (self.src_dir, self.umake_file), 'w') as make_hdl:
            make_hdl.write (make_src)


    def ucompile (self):
        """
        Compiles the translated code to a shared library, ready to be used.-
        """
        from os         import path, getcwd, chdir
        from ctypes     import cdll
        from subprocess import check_call

        try:
            #
            # start the compilation of the dynamic library
            #
            current_dir = getcwd ( )
            chdir (self.src_dir)
            check_call (["make", 
                         "--silent", 
                         "--file=%s" % self.umake_file])
            chdir (current_dir)
            #
            # attach the library object
            #
            self.ulib_obj = cdll.LoadLibrary ("%s" % path.join (self.src_dir, 
                                                                self.ulib_file))
        except Exception as e:
            logging.error ("Compilation error")
            self.ulib_obj = None
            raise e


    def is_valid_float_type_size (self, npfloat):
        rv = True

        if self.ulib_obj is None:
            self.generate_validation_code ( )
            self.ucompile ( )

        backendSize = self.ulib_obj.getFloatSize ( )
        nptype      = npfloat.dtype

        logging.debug ("Backend Float Size: %d" % backendSize)
        logging.debug ("Frontend NumPy Float Type: %s" % nptype)
    
        if nptype == np.float64:
            if backendSize != 64:
                rv = False                  # Floating point type precision mismatch!!!
        elif nptype == np.float32: 
            if backendSize != 32:
                rv = False                  # Floating point type precision mismatch!!!
        else:
            raise TypeError ("NumPy array element type (%s) does not match backend" % nptype)

        return rv

