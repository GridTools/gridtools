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

#include "stencil-composition/backend_metafunctions_fwd.hpp"

namespace gridtools {

    template < enumtype::platform BackendId, enumtype::grid_type GridId, enumtype::strategy StrategyType >
    struct is_backend< backend< BackendId, GridId, StrategyType > > : boost::mpl::true_ {};

    template < typename T >
    struct backend_id;

    template < enumtype::platform BackendId, enumtype::grid_type GridId, enumtype::strategy StrategyType >
    struct backend_id< backend< BackendId, GridId, StrategyType > >
        : enumtype::enum_type< enumtype::platform, BackendId > {};

} // gridtools
