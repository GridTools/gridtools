/*
  GridTools Libraries

  Copyright (c) 2016, GridTools Consortium
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  1. Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

  For information: http://eth-cscs.github.io/gridtools/
*/
#include <iostream>
#include <boost/numpy.hpp>

#include "IJKSizeConverter.hpp"
#include "DataFieldRepositoryPy.hpp"

#include "Coriolis.hpp"
#include "HorizontalAdvection.hpp"



namespace bp = boost::python;
namespace np = boost::numpy;



void initialize_module ( )
{
    //
    // initialize the NumPy backend
    //
    np::initialize ( );

    //
    // initialize the STELLA backend
    //
    std::cout << "Initializing the STELLA backend ..." << std::endl;

    //
    // register the Python <-> C++ converters
    //
    IJKSizeConverter ( );
}


BOOST_PYTHON_MODULE (_backend)
{
    //
    // package initialization
    //
    initialize_module ( );

    //
    // 3D calculation domain
    //
    bp::class_<IJKSize> ("IJKSize")
        .def ("get_i",
              &IJKSize::iSize)
        .def ("get_j",
              &IJKSize::jSize)
        .def ("get_k",
              &IJKSize::kSize)
    ;
    //
    // the DataFieldRepository contains all the structures needed to execute
    // the exposed stencils
    //
    bp::class_<DataFieldRepositoryPy, boost::noncopyable> ("DataFieldRepositoryPy")
        .def ("init_field",
              &DataFieldRepositoryPy::init_field)
        .def ("get_1d_field",
              &DataFieldRepositoryPy::get_1d_field)
        .def ("get_2d_field",
              &DataFieldRepositoryPy::get_2d_field)
        .def ("get_3d_field",
              &DataFieldRepositoryPy::get_3d_field)
    ;
    //
    // register shared pointers to different C++ types as valid Python objects;
    // the registration also prevents Python from copying, creating (i.e. no_init),
    // and destroying objects of these classes, since their memory management
    // is controlled from C++
    //
    bp::class_<JRealField, boost::shared_ptr<JRealField>, boost::noncopyable> ("JRealField", bp::no_init);
    bp::class_<IJRealField, boost::shared_ptr<IJRealField>, boost::noncopyable> ("IJRealField", bp::no_init);
    bp::class_<IJKRealField, boost::shared_ptr<IJKRealField>, boost::noncopyable> ("IJKRealField", bp::no_init);

    //
    // stencils exported from the backend, which are used by the proxy
    // classes in the 'stella.stencils' Python module.
    // They are easily recognizable because they have the same name.
    //

    //
    // select the correct version of the overloaded function ...
    //
    void (Coriolis::*CoriolisInit)(IJKRealField&,
                                   IJKRealField&,
                                   IJKRealField&,
                                   IJKRealField&,
                                   IJRealField&) = &Coriolis::Init;
    //
    // ... before exposing it
    //
    bp::class_<Coriolis, boost::noncopyable> ("Coriolis",
                                              "Provides access to the C++ stencil object.\n"
                                              "WARNING: This object is NOT meant to be used directly.")
        .def ("init",
              CoriolisInit)
        .def ("apply",
              &Coriolis::Do)
    ;
    //
    // select the correct version of the overloaded function ...
    //
    void (HorizontalAdvectionUV::*HorizontalAdvectionUVInit)(IJKRealField&,
                                                             IJKRealField&,
                                                             IJKRealField&,
                                                             IJKRealField&,
                                                             JRealField&,
                                                             JRealField&,
                                                             JRealField&,
                                                             const Real&,
                                                             const Real&) = &HorizontalAdvectionUV::Init;
    //
    // ... before exposing it
    //
    bp::class_<HorizontalAdvectionUV, boost::noncopyable> ("HorizontalAdvectionUV",
                                                           "Provides access to the C++ stencil object.\n"
                                                           "WARNING: this object is NOT meant to be used directly.")
        .def ("init",
              HorizontalAdvectionUVInit)
        .def ("apply",
              &HorizontalAdvectionUV::Apply)
    ;
}
