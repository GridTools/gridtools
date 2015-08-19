# -*- coding: utf-8 -*-
import logging

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


    def check_env_var (self):
        """
        Check if environment variables are correctly set
        """
        import os
        from gridtools import BASIC_ENVVARS
        from gridtools import CUDA_ENVVARS

        # Defining a list to gather the env vars not defined 
        missing_envvar = []

        # Using an indicator to identify the error type
        ind = 0
        for var in CUDA_ENVVARS:
           # Check if variables have been defined
           if os.getenv(var) is None:
               missing_envvar.append(var)             
               ind = 1
           else:
               # Check if path pointed by var exists 
               if not os.path.exists(os.getenv(var)):
                   ind=2

        msg="\n   Before continue: \n   1. Set variables "+str(BASIC_ENVVARS)+" for C++ backend and "+str(CUDA_ENVVARS)+" for CUDA backend \n   "
        if ind == 1:
          msg=msg+"2. Set the following undefined environment variable(s): "+str(missing_envvar)+" \n   EXIT NOW"

        if ind != 0:
           raise SystemError (msg)
 

