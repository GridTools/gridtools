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

#ifdef STRUCTURED_GRIDS
#include "stencil_composition/structured_grids/accessor_fwd.hpp"
#else
#include "stencil_composition/icosahedral_grids/accessor_fwd.hpp"
#endif

namespace gridtools {
    template < typename T >
    struct is_accessor;

    /**
     * Struct to test if an argument is a temporary
     */
    template < typename T >
    struct is_plchldr_to_temp;
}
