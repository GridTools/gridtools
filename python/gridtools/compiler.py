# -*- coding: utf-8 -*-
import logging

import numpy as np

from gridtools.utils import Utilities




class StencilCompiler ( ):
    """
    A global class that takes care of compiling the defined stencils 
    using different backends.-
    """
    def _initialize (self):
        """
        Initializes this Stencil compiler.-
        """
        from tempfile import mkdtemp

        logging.debug ("Initializing dynamic compiler ...")
        self.src_dir = mkdtemp (prefix="__gridtools_")
        self.utils.initialize ( )


    def __init__ (self):

        #
        # a dictionary containing the defined stencils (k=id(object), v=object)
        #
        self.stencils     = dict ( )
        self.lib_file     = "libgridtools4py"
        self.make_file    = "Makefile"
        #
        # these entities are automatically generated at compile time
        #
        self.src_dir      = None
        self.cpp_file     = None
        self.fun_hdr_file = None
        #
        # a reference to the compiled dynamic library
        #
        self.lib_obj      = None
        #
        # a utilities class for this compiler
        #
        self.utils        = Utilities (self)
        self._initialize ( )


    def compile (self, stencil):
        """
        Compiles the translated code to a shared library, ready to be used.-
        """
        from os                        import path, getcwd, chdir
        from ctypes                    import CDLL
        from subprocess                import check_call
        from numpy.distutils.misc_util import get_shared_lib_extension

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
            self.lib_obj = CDLL ('%s/%s%s' % (self.src_dir,
                                              self.lib_file,
                                              get_shared_lib_extension ( )))
        except Exception as e:
            logging.error ("Error while compiling '%s'" % stencil.name)
            self.lib_obj = None
            raise e


    def generate_code (self, stencil):
        """
        Generates native code for the received stencil:

            stencil     stencil object for which the code whould be generated.-
        """
        from os        import write, path, makedirs
        from gridtools import JinjaEnv

        try:
            #
            # create directory and files for the generated code
            #
            if not path.exists (self.src_dir):
                makedirs (self.src_dir)

            if stencil.backend == 'c++':
                extension = 'cpp'
            elif stencil.backend == 'cuda':
                extension = 'cu'
            else:
                raise RuntimeError ("Unknown backend '%s' in while generating code" % stencil.backend)
            self.cpp_file     = '%s.%s'    % (stencil.name, extension)
            self.fun_hdr_file = '%s_Functors.h' % stencil.name

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
            self.lib_obj      = None
            stencil.inspector = StencilInspector (stencil)


    def register (self, stencil):
        """
        Registers the received Stencil object `stencil` with the compiler.
        It returns a unique name for `stencil`.-
        """
        #
        # mark this stencil for recompilation ...
        #
        self.recompile (stencil)
        #
        # ... and add it to the registry if it is not there yet
        #
        if id(stencil) not in self.stencils.keys ( ):
            #
            # a unique name for this stencil object
            #
            stencil.name = '%s_%03d' % (stencil.__class__.__name__.capitalize ( ),
                                        len (self.stencils))
            self.stencils[id(stencil)] = stencil
        return stencil.name


    def run_native (self, stencil, **kwargs):
        """
        Executes of the received `stencil`.-
        """
        import ctypes

        #
        # make sure the stencil is registered
        #
        if id(stencil) not in self.stencils.keys ( ):
            self.register (stencil)

        #
        # run the selected backend version
        #
        if stencil.backend == 'c++' or stencil.backend == 'cuda':
            #
            # compile only if the library is not available
            #
            if self.lib_obj is None:
                stencil.resolve (**kwargs)
                self.generate_code (stencil)
                self.compile (stencil)
                #
                # floating point precision validation
                #
                for key in kwargs:
                    if isinstance(kwargs[key], np.ndarray):
                        if not self.utils.is_valid_float_type_size (kwargs[key]):
                            raise TypeError ("Element size of '%s' does not match that of the C++ backend."
                                              % key)
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

        functs               = dict ( )
        functs[stencil.name] = stencil.inspector.functors
        #
        # render the source code for each of the functors
        #
        functor_src = ""
        for f in functs[stencil.name]:
            functor_src += f.translate ( )
        #
        # instantiate each of the templates and render them
        #
        cpp    = JinjaEnv.get_template ("stencil.cpp")
        make   = JinjaEnv.get_template ("Makefile.cuda")

        params = list (stencil.scope.get_parameters ( ))
        temps  = list (stencil.scope.get_temporaries ( ))

        #
        # make sure the last stage is not independent
        #
        functs[stencil.name][-1].independent = False

        #
        # indices of the independent stencils needed to generate C++ code blocks
        #
        ind_funct_idx = list ( )
        for i in range (1, len (functs[stencil.name])):
            f = functs[stencil.name][i]
            if not f.independent:
                if functs[stencil.name][i - 1].independent:
                    ind_funct_idx.append (i - 1)

        return (functor_src,
                cpp.render (fun_hdr_file          = self.fun_hdr_file,
                            stencil_name          = stencil.name,
                            stencils              = [stencil],
                            scope                 = stencil.scope,
                            params                = params,
                            temps                 = temps,
                            params_temps          = params + temps,
                            functors              = functs,
                            independent_funct_idx = ind_funct_idx),
                make.render (stencils = [s for s in self.stencils.values ( ) if s.backend in ['c++', 'cuda']],
                             compiler = self))

