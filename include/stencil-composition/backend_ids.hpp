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

namespace gridtools {

    /**
     * @brief metadata with the information for architecture, grid and strategy backends
     * @tparam BackendId architecture backend id
     * @tparam GridId grid backend id
     * @tparam StrategyId strategy id
     */
    template < enumtype::platform BackendId, enumtype::grid_type GridId, enumtype::strategy StrategyId >
    struct backend_ids {
        static const enumtype::strategy s_strategy_id = StrategyId;
        static const enumtype::platform s_backend_id = BackendId;
        static const enumtype::grid_type s_grid_type_id = GridId;
    };

    template < typename T >
    struct is_backend_ids : boost::mpl::false_ {};

    template < enumtype::platform BackendId, enumtype::grid_type GridId, enumtype::strategy StrategyId >
    struct is_backend_ids< backend_ids< BackendId, GridId, StrategyId > > : boost::mpl::true_ {};
}
