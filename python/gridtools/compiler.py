# -*- coding: utf-8 -*-
import ast
import logging

import numpy as np

from gridtools.utils import Utilities



class StencilCompiler ( ):
    """
    A global class that takes care of compiling the defined stencils
    using different backends.-
    """
    BASE_LIB_NAME = "libgridtools4py"

    def __init__ (self):
        #
        # a dictionary containing the defined stencils (k=id(stencil), v=stencil)
        #
        self.stencils      = dict ( )
        self.lib_file      = None
        self.make_file     = "Makefile"
        #
        # these entities are automatically generated at compile time
        #
        self.src_dir       = None
        self.cpp_file      = None
        self.stg_hdr_file  = None
        #
        # a reference to the compiled dynamic library
        #
        self.lib_handle    = None
        #
        # track of the number of compilations
        #
        self.compile_count = 0
        #
        # an object to inspect the source code of the stencils
        #
        self.inspector     = StencilInspector ( )
        #
        # a utilities class for this compiler
        #
        self.utils         = Utilities (self)
        self._initialize ( )

    def __contains__ (self, stencil):
        """
        Returns True if the received stencils has been registered with this
        compiler
        :param stencil: the stencil object to look up
        :returns:       True if the stencil has been registered; False otherwise
        """
        return self.is_registered (stencil)


    def _initialize (self):
        """
        Initializes this Stencil compiler.-
        """
        from tempfile import mkdtemp

        logging.debug ("Initializing dynamic compiler ...")
        self.src_dir = mkdtemp (prefix="__gridtools_")
        self.utils.initialize ( )


    def analyze (self, stencil, **kwargs):
        """
        Performs a different analyses over the source code of the stencil
        :param stencil:      the stencil on which the static analysis should be
                             performed
        :param kwargs:       the parameters passed to this stencil for execution
        :raise LookupError:  if the stencil has not been registered with this
                             Compiler
        :raise NameError:    if no stencil stages could be extracted from the
                             source
        :raise RuntimeError: if the stencil's source code is not available,
                             e.g., if running from an interactive session
        :raise ValueError:   if the last stage is independent, which is an
                             invalid stencil
        :return:
        """
        if stencil in self:
            #
            # do not source-code analysis twice over the same code
            #
            if len (stencil.stages) == 0:
                #
                # try to resolve symbols by applying static-code analysis, ...
                #
                self.inspector.static_analysis (stencil)
                #
                # ... and by including runtime information
                #
                stencil.scope.runtime_analysis      (stencil, **kwargs)
                stencil.generate_code               ( )
                #
                # build and check stencil data dependency graph
                #
                stencil.build_data_dependency       ( )
                stencil.scope.check_data_dependency ( )
                #
                # build the stage-execution path
                #
                stencil.identify_stages_IO               ( )
                stencil.scope.build_execution_path       (stencil.name)
                stencil.scope.check_stage_execution_path ( )
                #
                # print out the discovered symbols if in DEBUG mode
                #
                if __debug__:
                    logging.debug ("Symbols found after applying runtime code analysis:")
                    stencil.scope.dump ( )
                    for stg in stencil.scope.stage_execution.nodes ( ):
                        stg.scope.dump ( )
            else:
                logging.info ("Not repeating source-code analysis of stencil '%s'" %
                              stencil.name)
        else:
            raise LookupError ("Stencil has not been registered with the compiler")


    def compile (self, stencil):
        """
        Compiles the translated code to a shared library, ready to be used.-
        """
        from os                        import getcwd, chdir
        from ctypes                    import CDLL
        from subprocess                import check_call
        from numpy.distutils.misc_util import get_shared_lib_extension

        try:
            #
            # start the compilation of the dynamic library
            #
            current_dir         = getcwd ( )
            chdir (self.src_dir)
            check_call (["make",
                         "--silent",
                         "--file=%s" % self.make_file])
            chdir (current_dir)
            #
            # attach the library object
            #
            self.lib_handle = CDLL ('%s/%s%s' % (self.src_dir,
                                                 self.lib_file,
                                                 get_shared_lib_extension ( )))

        except Exception as e:
            logging.error ("Error while compiling '%s'" % stencil.name)
            self.lib_handle = None
            raise e


    def generate_code (self, stencil):
        """
        Generates native code for the received stencil
        :param stencil: stencil object for which the code whould be generated
        :return:
        """
        from os        import path, makedirs
        from gridtools import JinjaEnv

        try:
            #
            # create directory and files for the generated code
            #
            self.compile_count += 1
            self.lib_file       = "%s.%04d" % (StencilCompiler.BASE_LIB_NAME,
                                               self.compile_count)
            if not path.exists (self.src_dir):
                makedirs (self.src_dir)
            if stencil.get_backend ( ) == 'cuda':
                extension = 'cu'
            else:
                extension = 'cpp'
            self.cpp_file     = '%s.%s'    % (stencil.name, extension)
            self.stg_hdr_file = '%s_Stages.h' % stencil.name
            #
            # ... and populate them
            #
            logging.info ("Generating %s code in '%s'" % (stencil.get_backend ( ).upper ( ),
                                                          self.src_dir))
            stg_src, cpp_src, make_src = self.translate (stencil)

            with open (path.join (self.src_dir, self.stg_hdr_file), 'w') as stg_hdl:
                stages = JinjaEnv.get_template ("stages.h")
                stg_hdl.write (stages.render (stage_src=stg_src))
            with open (path.join (self.src_dir, self.cpp_file), 'w') as cpp_hdl:
                cpp_hdl.write (cpp_src)
            with open (path.join (self.src_dir, self.make_file), 'w') as make_hdl:
                make_hdl.write (make_src)
        except Exception as e:
            logging.error ("Error while generating code:\n\t%s" % str (e))
            raise e


    def is_registered (self, stencil):
        """
        Checks whether a stencil is registered with this compiler
        :param stencil: the stencil to check for
        :return:        True if the stencil has been registered, False otherwise
        """
        return id (stencil) in self.stencils.keys ( )


    def recompile (self):
        """
        Marks the received stencil as dirty, needing recompilation.-
        """
        import _ctypes

        #
        # this only works in POSIX systems ...
        #
        if self.lib_handle is not None:
            _ctypes.dlclose (self.lib_handle._handle)
            del self.lib_handle
            self.lib_handle = None


    def register (self, stencil):
        """
        Registers the received Stencil object with this compiler
        :param stencil:   the stencil object to register
        :raise TypeError: in case the stencil object does not extend
                          MultiStageStencil
        :return:          a unique name for the given stencil
        """
        from gridtools.stencil import MultiStageStencil

        if issubclass (stencil.__class__, MultiStageStencil):
            #
            # mark this stencil for recompilation ...
            #
            self.recompile ( )
            #
            # ... and add it to the registry if it is not there yet
            #
            if stencil not in self:
                #
                # a unique name for this stencil object
                #
                stencil.name = '%s_%03d' % (stencil.__class__.__name__.capitalize ( ),
                                            len (self.stencils))
                self.stencils[id(stencil)] = stencil
                logging.debug ("Stencil '%s' registered with the Compiler" % stencil.name)
            return stencil.name
        else:
            raise TypeError ("Stencil should inherit from MultiStageStencil")


    def run_native (self, stencil, **kwargs):
        """
        Compiles and executes of the stencil in native mode
        :param stencil: the stencil to be executed
        :return:
        """
        import ctypes
        #
        # compile only if the library is not available
        #
        if self.lib_handle is None:
            self.generate_code (stencil)
            self.compile       (stencil)
            #
            # Array validation (floating point precision, memory layout...)
            #
            for key in kwargs:
                if isinstance(kwargs[key], np.ndarray):
                    if not self.utils.is_valid_float_type_size (kwargs[key]):
                        raise TypeError ("Element size of '%s' does not match \
                                          that of the C++ backend." % key)
                    kwargs[key] = self.utils.enforce_optimal_array (kwargs[key],
                                                                    key,
                                                                    stencil.get_backend ( ))
        #
        # prepare the list of parameters to call the library function
        #
        lib_params = list (stencil.domain)
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
        run_func = self.lib_handle['run_%s' % stencil.name]
        run_func (*lib_params)


    def translate (self, stencil):
        """
        Translates the received stencil to C++, using the gridtools interface,
        returning a string tuple of rendered files (stages, cpp, make).-
        """
        from gridtools import JinjaEnv

        stgs               = dict ( )
        stgs[stencil.name] = list (stencil.stages)

        #
        # render the source code for each of the stages
        #
        stage_src = ""
        for f in stgs[stencil.name]:
            stage_src += f.translate ( )
        #
        # instantiate each of the templates and render them
        #
        cpp    = JinjaEnv.get_template ("stencil.cpp")
        make   = JinjaEnv.get_template ("Makefile.cuda")

        params = list (stencil.scope.get_parameters ( ))
        temps  = list (stencil.scope.get_temporaries ( ))

        #
        # indices of the independent stencils needed to generate C++ code blocks
        #
        ind_stg_idx = list ( )
        for i in range (1, len (stgs[stencil.name])):
            f = stgs[stencil.name][i]
            if not f.independent:
                if stgs[stencil.name][i - 1].independent:
                    ind_stg_idx.append (i - 1)

        return (stage_src,
                cpp.render (stg_hdr_file          = self.stg_hdr_file,
                            stencil_name          = stencil.name,
                            stencils              = [stencil],
                            scope                 = stencil.scope,
                            params                = params,
                            temps                 = temps,
                            params_temps          = params + temps,
                            stages                = stgs,
                            independent_stage_idx = ind_stg_idx),
                make.render (stencils = [s for s in self.stencils.values ( ) if s.get_backend ( ) in ['c++', 'cuda']],
                             compiler = self))

    def unregister (self, stencil):
        """
        Removes registration of the received Stencil object from this compiler
        :param stencil:   the stencil object to unregister
        """
        if self.is_registered (stencil):
            #
            # Remove this stencil from the compiler registry
            #
            del self.stencils[id(stencil)]
            logging.debug ("Stencil '%s' unregistered from the Compiler" % stencil.name)
        else:
            logging.warning("Trying to unregister Stencil '%s' that is not \
                             registered with the Compiler")




class StencilInspector (ast.NodeVisitor):
    """
    Inspects the source code of a stencil definition using its AST.-
    """
    def __init__ (self):
        super ( ).__init__ ( )
        #
        # a reference to the currently inspected stencil, needed because the
        # NodeVisitor pattern does not allow extra parameters
        #
        self.inspected_stencil = None
        #
        # stage definitions are kept here as they are discovered in the source
        #
        self.stage_defs      = list ( )


    def _analyze_params (self, nodes):
        """
        Extracts the stencil parameters from an AST-node list
        :param nodes: the node list from which the parameters should be extracted
        """
        for n in nodes:
            #
            # do not add the 'self' parameter
            #
            if n.arg != 'self':
                #
                # parameters starting with the 'in_' prefix are considered 'read only'
                #
                read_only = n.arg.startswith ('in_')
                self.inspected_stencil.scope.add_parameter (n.arg,
                                                            read_only=read_only)


    def _check_kernel_decorator (self, node):
        """
        Checks if the given AST node has been decorated with the
        stencil_kernel decorator, which identifies it as the stencil entry point
        :param node: The AST node to be checked
        :return:     True if the node represents a function definition decorated
                     as the kernel, False otherwise
        """
        #
        # Only FunctionDef nodes have decorator_lists, and we only want non-empty
        # lists
        #
        if not isinstance (node, ast.FunctionDef) or not node.decorator_list:
            return False
        #
        # The decorator must be an Attribute AST node, with value id 'Stencil'
        # and attribute name 'kernel'
        #
        return any (isinstance(x, ast.Attribute)
                    and x.value.id == 'Stencil'
                    and x.attr == 'kernel'
                    for x in node.decorator_list)


    def _extract_source (self):
        """
        Extracts the source code from the currently inspected stencil
        """
        import inspect
        from gridtools import STENCIL_KERNEL_DECORATOR_LABEL

        src = 'class %s (%s):\n' % (str (self.inspected_stencil.__class__.__name__),
                                    str (self.inspected_stencil.__class__.__base__.__name__))
        #
        # first the constructor and stages
        #
        for (name,fun) in inspect.getmembers (self.inspected_stencil,
                                              predicate=inspect.ismethod):
            try:
                if name == '__init__' or name.startswith ('stage_'):
                    src += inspect.getsource (fun)
            except OSError:
                try:
                    #
                    # is this maybe a notebook session?
                    #
                    from IPython.code import oinspect
                    src += oinspect.getsource (fun)
                except Exception:
                    raise RuntimeError ("Could not extract source code from '%s'"
                                        % self.inspected_stencil.__class__)
        #
        # then the kernel, which lies inside the kernel_wrapper
        #
        kernel_found = False
        for (name,fun) in inspect.getmembers (self.inspected_stencil,
                                              predicate=inspect.ismethod):
            try:
                #
                # To identify the kernel wrapper, we check the attribute set by
                # the decorator
                #
                if hasattr (fun, STENCIL_KERNEL_DECORATOR_LABEL):
                    if not kernel_found:
                        kernel_found = True
                    else:
                        #
                        # There can be only one stencil kernel
                        #
                        raise AttributeError ("Multiple kernels detected for\
                                              stencil %s. Please define only a\
                                              single kernel."
                                              % self.inspected_stencil.__class__)
                    #
                    # Since the stencil_kernel decorator uses functools' @wraps, we
                    # know that the kernel can be found inside the __wrapped__
                    # attribute of the wrapper.
                    # Another way could be to use inspect.unwrap(fun) to directly
                    # get the kernel function object.
                    #
                    src += inspect.getsource (fun.__wrapped__)
            except OSError:
                try:
                    #
                    # is this maybe a notebook session?
                    #
                    from IPython.code import oinspect
                    src += oinspect.getsource (fun.__wrapped__)
                except Exception:
                    raise RuntimeError ("Could not extract source code from '%s'"
                                        % self.inspected_stencil.__class__)
        #
        # Raise an AttributeError if a kernel could not be found
        #
        if not kernel_found:
            raise AttributeError ("No kernel detected for stencil %s! Please \
                                  define a stencil entry point function."
                                  % self.inspected_stencil.__class__)
        return src


    def static_analysis (self, stencil):
        """
        Performs a static analysis over the source code of the received stencil
        :param stencil:      the stencil on which the static analysis should be
                             performed
        :raise NameError:    if no stencil stages could be extracted from the
                             source
        :raise RuntimeError: if the stencil's source code is not available,
                             e.g., if running from an interactive session
        :return:
        """
        try:
            assert (self.inspected_stencil is None), "Trying to start static code analysis with `inspected_stencil` already set"
            #
            # initialize the state variables
            #
            self.inspected_stencil = stencil
            self.stage_defs        = list ( )
            st                     = self.inspected_stencil

            if st.scope.py_src is None:
                st.scope.py_src = self._extract_source ( )

            if st.scope.py_src is not None:
                self.ast_root = ast.parse (st.scope.py_src)
                self.visit (self.ast_root)
                #
                # print out the discovered symbols if in DEBUG mode
                #
                if __debug__:
                    logging.debug ("Symbols found after static code analysis:")
                    st.scope.dump ( )
                if len (st.stages) == 0:
                    raise NameError ("Could not extract any stage from stencil '%s'" % st.name)
            else:
                #
                # if the source code is still not available, we may infer
                # the user is running from some weird interactive session
                #
                raise RuntimeError ("Source code not available.\nSave your stencil class(es) to a file and try again.")
        except Exception as e:
            self.inspected_stencil = None
            raise e
        else:
            self.inspected_stencil = None


    def visit_Assign (self, node):
        """
        Extracts symbols appearing in assignments in the user's stencil code
        :param node:         a node from the AST
        :raise RuntimeError: if more than one assignment per line is found
        :return:
        """
        #
        # expr = expr
        #
        if len (node.targets) > 1:
            raise RuntimeError ("Only one assignment per line is accepted.")
        else:
            st          = self.inspected_stencil
            lvalue      = None
            lvalue_node = node.targets[0]
            #
            # attribute assignment
            #
            if isinstance (lvalue_node, ast.Attribute):
                lvalue = "%s.%s" % (lvalue_node.value.id,
                                    lvalue_node.attr)
            #
            # parameter or local variable assignment
            #
            elif isinstance (lvalue_node, ast.Name):
                lvalue = lvalue_node.id
            else:
                logging.debug ("Ignoring assignment at %d" % node.lineno)
                return

            rvalue_node = node.value
            #
            # a constant if its rvalue is a Num
            #
            if isinstance (rvalue_node, ast.Num):
                rvalue = float (rvalue_node.n)
                st.scope.add_constant (lvalue, rvalue)
                logging.debug ("Adding numeric constant '%s'" % lvalue)
            #
            # variable names are resolved using runtime information
            #
            elif isinstance (rvalue_node, ast.Name):
                try:
                    rvalue = eval (rvalue_node.id)
                    st.scope.add_constant (lvalue, rvalue)
                    logging.debug ("Adding constant '%s'" % lvalue)

                except NameError:
                    st.scope.add_constant (lvalue, None)
                    logging.debug ("Delayed resolution of constant '%s'" % lvalue)
            #
            # function calls are resolved later by name
            #
            elif isinstance (rvalue_node, ast.Call):
                rvalue = None
                st.scope.add_constant (lvalue, rvalue)
                logging.debug ("Constant '%s' holds a function value" % lvalue)
            #
            # attributes are resolved using runtime information
            #
            elif isinstance (rvalue_node, ast.Attribute):
                rvalue = getattr (eval (rvalue_node.value.id),
                                  rvalue_node.attr)
                st.scope.add_constant (lvalue, rvalue)
                logging.debug ("Constant '%s' holds an attribute value" % lvalue)
            #
            # try to discover the correct type using runtime information
            #
            else:
                #
                # we keep all other expressions and try to resolve them later
                #
                st.scope.add_constant (lvalue, None)
                logging.debug ("Constant '%s' will be resolved later" % lvalue)


    def visit_Expr (self, node):
        """
        Looks for named stages within a stencil
        :param node:      a node from the AST
        :raise TypeError: if the type of a keyword argument cannot be infered
        :return:
        """
        if isinstance (node.value, ast.Call):
            call = node.value
            if (isinstance (call.func, ast.Attribute) and
                isinstance (call.func.value, ast.Name)):
                if (call.func.value.id == 'self' and
                    call.func.attr.startswith ('stage_') ):
                    #
                    # found a new independent stage
                    #
                    stage       = None
                    name_suffix = call.func.attr
                    #
                    # look for its definition
                    #
                    for stg_def in self.stage_defs:
                        if stg_def.name == call.func.attr:
                            for node in stg_def.body:
                                if isinstance (node, ast.For):
                                    stage = self.visit_For (node,
                                                            name_suffix=name_suffix)
                    assert (stage is not None)
                    #
                    # extract its parameters
                    #
                    if len (call.args) > 0:
                        logging.warning ("Ignoring positional arguments when calling intermediate stages")
                    else:
                        for kw in call.keywords:
                            if isinstance (kw.value, ast.Attribute):
                                stage.scope.add_alias (kw.arg,
                                                       '%s.%s' % (kw.value.value.id,
                                                                  kw.value.attr))
                            elif isinstance (kw.value, ast.Name):
                                stage.scope.add_alias (kw.arg,
                                                       kw.value.id)
                            else:
                                raise TypeError ("Unknown type '%s' of keyword argument '%s'"
                                                 % (kw.value.__class__, kw.arg))


    def visit_For (self, node, name_suffix=None):
        """
        Looks for 'get_interior_points' comprehensions
        :param node:        a node from the AST
        :param name_suffix: if given, this value is passed as a suffix of the
                            stage name
        :return:            the created Stage object
        """
        #
        # the iteration should call 'get_interior_points'
        #
        st    = self.inspected_stencil
        call  = node.iter
        stage = None

        if isinstance (call.func, ast.Attribute):
            if (call.func.value.id in ['Stencil', 'self']
                and call.func.attr == 'get_interior_points'):
                if name_suffix is None:
                    stage = st.scope.add_stage (node,
                                                prefix=st.name.lower ( ),
                                                suffix='stage')
                else:
                    #
                    # the suffix is present only for independent stages
                    #
                    stage = st.scope.add_stage (node,
                                                prefix=st.name.lower ( ),
                                                suffix=name_suffix)

        return stage


    def visit_FunctionDef (self, node):
        """
        Looks for function definitions inside the user's stencil and classifies
        them accordingly
        :param node:         a node from the AST
        :raise RuntimeError: if the parent constructor call is missing from the
                             user's defined stencil
        :raise ValueError:   if the kernel function returns anything other than
                             None
        :return:
        """
        #
        # the stencil constructor is the recommended place to define
        # (pre-calculated) constants and temporary data fields
        #
        if node.name == '__init__':
            logging.debug ("Stencil constructor found")
            docstring = ast.get_docstring(node)
            #
            # should be a call to the parent-class constructor
            #
            pcix = 0 # Index, amongst the children nodes, of the call to parent constructor
            for n in node.body:
                if isinstance(n.value, ast.Str):
                    # Allow for the docstring to appear before the call to the parent constructor
                    if n.value.s.lstrip() != docstring:
                        pcix = pcix + 1

                else:
                    pcix = pcix + 1
                try:
                    parent_call = (isinstance (n.value, ast.Call) and
                                   isinstance (n.value.func.value, ast.Call) and
                                   n.value.func.attr == '__init__')
                    if parent_call:
                        logging.debug ("Parent constructor call found")
                        break
                except AttributeError:
                    parent_call = False
            #
            # inform the user if the call was not found
            #
            if not parent_call:
                raise RuntimeError ("Missing parent constructor call")
            if pcix != 1:
                raise RuntimeError ("Parent constructor is NOT the first operation of the child constructor")
            #
            # continue traversing the AST of this function
            #
            for n in node.body:
                self.visit (n)
        #
        # The kernel function is the starting point of the stencil.
        # We can identify its AST Node by checking its decorators
        #
        elif self._check_kernel_decorator (node):
            logging.debug ("Entry function '%s' found at line %d" % (node.name, node.lineno))
            #
            # this function should return 'None'
            #
            for n in node.body:
                if isinstance (n, ast.Return) and n.value is not None:
                    raise ValueError ("The kernel function should return 'None'")
            #
            # the parameters of the kernel function are the stencil
            # arguments in the generated code
            #
            self._analyze_params (node.args.args)
            #
            # Store the name of the kernel function in the stencil
            #
            self.inspected_stencil.entry_point_name = node.name
            #
            # continue traversing the AST
            #
            for n in node.body:
                self.visit (n)
        #
        # other function definitions are saved for potential use later
        #
        else:
            self.stage_defs.append (node)
