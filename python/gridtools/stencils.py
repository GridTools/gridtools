import numpy as np

import stella



class Stencil (object):
    """
    A base class for all exposed stencils.-
    """
    def _check_params (self, params):
        """
        Checks the received iterable for NumPy arrays with the expected
        memory layout and element types.-
        """
        for arr in params:
            try:
                if arr.dtype != np.float64:
                    raise Warning ("The array should contain double-precision floats.")
                #
                # for an explanation on NumPy array flags, see 
                # http://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.flags.html#numpy.ndarray.flags
                #
                if not arr.flags.f_contiguous:
                    raise Warning ("The array should be in Fortran-contiguous format.")

            except AttributeError:
                #
                # this parameter is not a NumPy array, ignore it
                #
                pass


    def __init__ (self):
        #
        # a member holding a reference to the used stencil
        # NOTE each inheriting stencil should instantiate a valid stencil object
        #
        self._stencil = None
        #
        # list with the accepted names of the input and output stencil parameters
        # NOTE each inheriting stencil should define these two tuples
        # NOTE order is important!! It should the C++ function arity
        #
        self._input_data_fields  = None
        self._output_data_fields = None
        #
        # initialize the module's data-field repository if not already
        #
        if stella.__repository__ is None:
            stella.__repository__ = stella.DataFieldRepository ( )


    def prepare (self, **kwargs):
        """
        Prepares the stencil for execution using 'kwargs' as the input data
        fields. It returns the stencil parameters, ready to be used.-
        """
        #
        # check that all stencil fields are available as parameters
        #
        if all (df in kwargs.keys ( ) for df in self._input_data_fields):
            #
            # get the data field arguments and initialize them
            #
            self._check_params (kwargs.values ( ))
            #
            # initialize the data fields
            #
            ret_value = list ( )
            for arg in self._input_data_fields:
                stella.__repository__.init_field (arg, kwargs[arg])
                ret_value.append (stella.__repository__.get_field (arg))
            #
            # append the non-data-fields to the final parameter list
            #
            for arg,val in kwargs.items ( ):
                if arg not in self._input_data_fields:
                    ret_value.append (val)
            #
            # issue a warning if a parameter hasn't been found
            #
            if None in ret_value:
                raise Warning ("There are missing parameters %s" % str (ret_value))
            return ret_value
        else:
            raise Warning ("Missing stencil parameters %s.\nAccepted are %s." % 
                           (str (kwargs.keys ( )),
                            str (self._input_data_fields)))



class Coriolis (Stencil):
    """
    The Coriolis force stencil.-
    """
    def __init__ (self):
        """
        Creates a new instance of a Coriolis stencil.-
        """
        super ( ).__init__ ( )
        self._stencil = stella._backend.Coriolis ( )
        self._input_data_fields = tuple ( ('utens',
                                           'vtens',
                                           'u_nnow',
                                           'v_nnow',
                                           'fc') )
        self._output_data_fields = tuple ( ('utens',
                                            'vtens') )

    def apply (self, **kwargs):
        """
        Applies the Coriolis stencil using the received NumPy arrays
        as named parameters:

            utens   a 3D data field;
            vtens   a 3D data field;
            u_nnow  a 3D data field;
            v_nnow  a 3D data field;
            fc      a 2D data field.-
        """
        params = self.prepare (**kwargs)
        self._stencil.init (*params)
        self._stencil.apply ( )



class HorizontalAdvectionUV (Stencil):
    """
    The Horizontal Advection stencil over U and V.-
    """
    def __init__ (self):
        """
        Creates a new instance of this stencil.-
        """
        super ( ).__init__ ( )
        self._stencil = stella._backend.HorizontalAdvectionUV ( )
        #
        # note that scalar arguments are not specified
        #
        self._input_data_fields = tuple ( ('utens_stage',
                                           'vtens_stage',
                                           'u_stage',
                                           'v_stage',
                                           'acrlat0',
                                           'acrlat1',
                                           'tgrlat') )
        self._output_data_fields = tuple ( ('utens_stage',
                                            'vtens_stage') )

    def apply (self, **kwargs):
        """
        Applies the stencil using the received NumPy arrays
        as named parameters:

            utens_stage     a 3D data field;
            vtens_stage     a 3D data field;
            u_stage         a 3D data field;
            v_stage         a 3D data field;
            acrlat0         a 1D data field;
            acrlat1         a 1D data field;
            tgrlat          a 1D data field;
            eddlat          a scalar;
            eddlon          a scalar.-
        """
        params = self.prepare (**kwargs)
        self._stencil.init (*params)
        self._stencil.apply ( )




"""
class HorizontalDiffusionUV (Stencil):
    ""
    Class holding the horizontal diffusion stencil for u and v
    ""
    def __init__ (self, dycore):
        ""
        Creates a new instance of this stencil using the data fields
        of the received Dycore for input and output:

        dycore  the Dycore object to use.-
        ""
        self._dycore  = dycore
        self._stencil = _backend.HorizontalDiffusionUV ( )
        self._stencil.init (self._dycore.communication_configuration,
                            self._dycore.repository,
                            self._dycore.horizontal_diffusion_repository)

    def apply_u (self):
        self._stencil.apply_u ( )

    def apply_v (self):
        self._stencil.apply_v ( )
"""
