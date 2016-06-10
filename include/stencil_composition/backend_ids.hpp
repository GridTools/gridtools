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
