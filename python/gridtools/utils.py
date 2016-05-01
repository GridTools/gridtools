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
        from os        import write, path
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


    def caller_name(skip=2):
        """Get a name of a caller in the format module.class.method

           `skip` specifies how many levels of stack to skip while getting caller
           name. skip=1 means "who calls me", skip=2 "who calls my caller" etc.

           An empty string is returned if skipped levels exceed stack height

           Taken from https://gist.github.com/techtonik/2151727
        """
        stack = inspect.stack()
        start = 0 + skip
        if len(stack) < start + 1:
          return ''
        parentframe = stack[start][0]

        name = []
        module = inspect.getmodule(parentframe)
        # `modname` can be None when frame is executed directly in console
        # TODO(techtonik): consider using __main__
        if module:
            name.append(module.__name__)
        # detect classname
        if 'self' in parentframe.f_locals:
            # I don't know any way to detect call from the object method
            # XXX: there seems to be no way to detect static method call - it will
            #      be just a function call
            name.append(parentframe.f_locals['self'].__class__.__name__)
#            print('parentframe class:',parentframe.f_locals['self'].__class__)
        codename = parentframe.f_code.co_name
        if codename != '<module>':  # top level usually
            name.append( codename ) # function or a method
        del parentframe
#        print('caller name list:',name)
        return ".".join(name)


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
        # TODO(techtonik): consider using __main__
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
#        print('caller_class', caller_class)
#        print('caller name:', caller_name)
#        print('stencil class:', stencil.__class__)
#        print('isinstance:',isinstance(stencil, caller_class))
#        print('caller_name == run',caller_name=='run')
        return isinstance(stencil, caller_class) and caller_name == 'run'
