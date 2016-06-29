/*
   Copyright 2016 GridTools Consortium

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
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
