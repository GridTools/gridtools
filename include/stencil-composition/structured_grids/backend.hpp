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

        static const enumtype::strategy s_strategy_id = base_t::s_strategy_id;
        static const enumtype::platform s_backend_id = base_t::s_backend_id;
        static const enumtype::grid_type s_grid_type_id = base_t::s_grid_type_id;
    };

} // namespace gridtools
