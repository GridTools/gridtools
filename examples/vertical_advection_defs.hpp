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
#pragma once
#include <gridtools.hpp>

namespace vertical_advection {

// define some physical constants
#define BETA_V ((double)0.0)
#define BET_M ((double)0.5 * ((double)1.0 - BETA_V))
#define BET_P ((double)0.5 * ((double)1.0 + BETA_V))

#ifdef CUDA_EXAMPLE
    typedef gridtools::backend< gridtools::enumtype::Cuda, gridtools::GRIDBACKEND, gridtools::enumtype::Block >
        va_backend;
#else
#ifdef BACKEND_BLOCK
    typedef gridtools::backend< gridtools::enumtype::Host, gridtools::GRIDBACKEND, gridtools::enumtype::Block >
        va_backend;
#else
    typedef gridtools::backend< gridtools::enumtype::Host, gridtools::GRIDBACKEND, gridtools::enumtype::Naive >
        va_backend;
#endif
#endif
}
