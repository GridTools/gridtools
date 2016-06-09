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

#include "stencil-composition/backend_base.hpp"

namespace gridtools {

    template < enumtype::platform BackendId, enumtype::strategy StrategyType >
    struct backend< BackendId, enumtype::structured, StrategyType >
        : public backend_base< BackendId, enumtype::structured, StrategyType > {
        typedef backend_base< BackendId, enumtype::structured, StrategyType > base_t;

        using typename base_t::backend_traits_t;
        using typename base_t::strategy_traits_t;
        using typename base_t::grid_traits_t;
    };

} // namespace gridtools
