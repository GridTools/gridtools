# -*- coding: utf-8 -*-
import logging
import inspect

import numpy as np



class Utilities ( ):
    """
    Class to contain various helpful functions.
    Currently contains floating point precision validation.-
    """
    def __init__ (self, compiler):
        """
        Creates a new Utilities class with a reference to the session's compiler.-
        """
        self.compiler  = compiler
        self.tmpl_file = 'Utilities.cpp'


    def initialize (self):
        """
        Generates native code for this utilities class.-
        """
        from os        import path
        from gridtools import JinjaEnv

        logging.debug ("Generating backend float type check code (C++) in '%s'" % self.compiler.src_dir)

        utils_tmpl = JinjaEnv.get_template (self.tmpl_file)
        utils_src  = utils_tmpl.render ( )

        with open (path.join (self.compiler.src_dir,
                              self.tmpl_file), 'w') as cpp_hdl:
            cpp_hdl.write (utils_src)


    def is_valid_float_type_size (self, npfloat):
        rv = True

        backendSize = self.compiler.lib_handle.get_backend_float_size ( )
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


    def is_fortran_array_layout (self, nparray):
        rv = False

        if nparray.flags['F_CONTIGUOUS']:
            logging.debug("Fortran array layout in use.")
            rv = True
        else:
            logging.info("Non Fortran array layout in use!!!")
            rv = False

        return rv
        

    def check_kernel_caller(stencil):
        """
        Check that the kernel function for the input stencil is being called
        by the run() method of the stencil class itself.
        In order to carry out its intended purpose, this function should only be
        used inside kernel wrapper functions.

        Modified from https://gist.github.com/techtonik/2151727

        :param stencil: The stencil object whose kernel is being called
        :return:        True if the kernel is being called from its own stencil
                        run() method, False otherwise
        """
        stack = inspect.stack()
        if len(stack) < 3:
          return False
        parentframe = stack[2][0]

        module = inspect.getmodule(parentframe)
        # `modname` can be None when frame is executed directly in console
        # TODO: consider using __main__
        if not module:
            return False

        #
        # Detect caller class
        #
        caller_class = None
        if 'self' in parentframe.f_locals:
            # XXX: there seems to be no way to detect static method call - it will
            #      be just a function call
            caller_class = parentframe.f_locals['self'].__class__

        #
        # Detect caller name
        #
        caller_name = parentframe.f_code.co_name
        if caller_name == '<module>':  # top level usually
            return False
        del parentframe

        return isinstance(stencil, caller_class) and caller_name == 'run'

