# -*- coding: utf-8 -*-
import logging

import numpy as np

from gridtools.utils   import Utilities




class StencilCompiler ( ):
    """
    A global class that takes care of compiling the defined stencils using different backends.-
    """
    #
    # a utilities class shared by all stencils
    #
    utils = Utilities ( )

    def __init__ (self):
        #
        # a dictionary containing all the defined stencils (k=name, v=object)
        #
        self.stencils = dict ( )
        #
        # these entities are automatically generated at compile time
        #
        self.src_dir      = None
        self.lib_file     = None
        self.cpp_file     = None
        self.make_file    = None
        self.fun_hdr_file = None
        #
        # a reference to the compiled dynamic library
        #
        self.lib_obj = None


    def compile (self, stencil):
        """
        Compiles the translated code to a shared library, ready to be used.-
        """
        from os         import path, getcwd, chdir
        from numpy      import ctypeslib
        from subprocess import check_call

        try:
            #
            # start the compilation of the dynamic library
            #
            current_dir = getcwd ( )
            chdir (self.src_dir)
            check_call (["make", 
                         "--silent", 
                         "--file=%s" % self.make_file])
            chdir (current_dir)
            #
            # attach the library object
            #
            self.lib_obj = ctypeslib.load_library (self.lib_file,
                                                   self.src_dir)
        except Exception as e:
            logging.error ("Error while compiling '%s'" % stencil.name)
            self.lib_obj = None
            raise e


    def generate_code (self, stencil, src_dir=None):
        """
        Generates native code for the received stencil:

            stencil     stencil object for which the code whould be generated;
            src_dir     directory where the files should be saved (optional).-
        """
        from os        import write, path, makedirs
        from tempfile  import mkdtemp
        from gridtools import JinjaEnv

        try:
            #
            # create directory and files for the generated code
            #
            if src_dir is None:
                self.src_dir = mkdtemp (prefix="__gridtools_")
            else:
                if not path.exists (src_dir):
                    makedirs (src_dir)
                self.src_dir = src_dir

            if stencil.backend == 'c++':
                extension = 'cpp'
            elif stencil.backend == 'cuda':
                extension = 'cu'
            else:
                raise RuntimeError ("Unknown backend '%s' in while generating code" % stencil.backend)
            self.cpp_file     = '%s.%s'    % (stencil.name, extension)
            self.lib_file     = 'lib%s'    % stencil.name.lower ( )
            self.make_file    = 'Makefile'
            self.fun_hdr_file = '%sFunctors.h' % stencil.name

            #
            # ... and populate them
            #
            logging.info ("Generating %s code in '%s'" % (stencil.backend.upper ( ),
                                                          self.src_dir))
            #
            # generate the code of *all* functors in this stencil,
            # build a data-dependency graph among *all* data fields
            #
            for func in stencil.inspector.functors:
                func.generate_code (stencil.inspector.src)
                stencil.scope.add_dependencies (func.get_dependency_graph ( ).edges ( ))
            fun_src, cpp_src, make_src = self.translate (stencil)


            with open (path.join (self.src_dir, self.fun_hdr_file), 'w') as fun_hdl:
                functors  = JinjaEnv.get_template ("functors.h")
                fun_hdl.write (functors.render (functor_src=fun_src))
            with open (path.join (self.src_dir, self.cpp_file), 'w') as cpp_hdl:
                cpp_hdl.write (cpp_src)
            with open (path.join (self.src_dir, self.make_file), 'w') as make_hdl:
                make_hdl.write (make_src)

        except Exception as e:
            logging.error ("Error while generating code:\n\t%s" % str (e))
            raise e


    def recompile (self, stencil):
        """
        Marks the received stencil as dirty, needing recompilation.-
        """
        import _ctypes
        from gridtools.stencil import StencilInspector

        #
        # this only works in POSIX systems ...
        #
        if self.lib_obj is not None:
            _ctypes.dlclose (self.lib_obj._handle)
            del self.lib_obj
            self.lib_obj   = None
            stencil.inspector = StencilInspector (stencil)


    def register (self, stencil):
        """
        Registers the received Stencil object `stencil_obj` with the compiler.
        It returns a unique name for `stencil_obj`.-
        """
        #
        # a unique name for this stencil object
        #
        name = '%s_%05d' % (stencil.__class__.__name__.capitalize ( ),
                            len (self.stencils))
        self.stencils[name] = stencil

        return name


    def run_native (self, stencil, **kwargs):
        """
        Executes of the received `stencil`.-
        """
        import ctypes

        #
        # run the selected backend version
        #
        if stencil.backend == 'c++' or stencil.backend == 'cuda':
            #
            # automatic compilation only if the library is not available
            #
            if self.lib_obj is None:
                #
                # floating point precision validation
                #
                for key in kwargs:
                      if isinstance(kwargs[key], np.ndarray):
                          if not StencilCompiler.utils.is_valid_float_type_size (kwargs[key]):
                              raise TypeError ("Element size of '%s' does not match that of the C++ backend."
                                               % key)
                stencil.resolve (**kwargs)
                self.generate_code (stencil)
                self.compile (stencil)
            #
            # prepare the list of parameters to call the library function
            #
            lib_params = list (stencil.inspector.domain)

            #
            # extract the buffer pointers from the NumPy arrays
            #
            for p in stencil.scope.get_parameters ( ):
                if p.name in kwargs.keys ( ):
                    lib_params.append (kwargs[p.name].ctypes.data_as (ctypes.c_void_p))
                else:
                    logging.warning ("Parameter '%s' does not exist in the symbols table" % p.name)
            #
            # call the compiled stencil
            #
            run_func = getattr (self.lib_obj, 'run_%s' % stencil.name)
            run_func (*lib_params)
        else:
            logging.error ("Unknown backend '%s'" % self.backend)


    def translate (self, stencil):
        """
        Translates the received stencil to C++, using the gridtools interface, 
        returning a string tuple of rendered files (functors, cpp, make).-
        """
        from gridtools import JinjaEnv

        #
        # render the source code for each of the functors
        #
        functor_src = ""
        for f in stencil.inspector.functors:
            functor_src += f.translate ( )
        #
        # instantiate each of the templates and render them
        #
        cpp    = JinjaEnv.get_template ("stencil.cpp")
        make   = JinjaEnv.get_template ("Makefile.%s" % stencil.backend)

        params = list (stencil.scope.get_parameters ( ))
        temps  = list (stencil.scope.get_temporaries ( ))

        functs     = dict ( )
        ind_functs = dict ( )

        functs[stencil.name]     = [f for f in stencil.inspector.functors if not f.independent]
        ind_functs[stencil.name] = [f for f in stencil.inspector.functors if f.independent]
       
        #
        # make sure there is at least one non-independent functor
        #
        if len (functs[stencil.name]) == 0:
            functs[stencil.name]     = [ stencil.inspector.functors[-1] ]
            ind_functs[stencil.name] = ind_functs[stencil.name][:-1]

        return (functor_src,
                cpp.render (fun_hdr_file         = self.fun_hdr_file,
                            stencil_name         = stencil.name,
                            stencils             = [stencil],
                            scope                = stencil.scope,
                            params               = params,
                            temps                = temps,
                            params_temps         = params + temps,
                            functors             = functs,
                            independent_functors = ind_functs),
                make.render (stencil=stencil))

