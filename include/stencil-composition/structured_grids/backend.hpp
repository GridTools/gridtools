#pragma once

#include "stencil-composition/backend_base.hpp"

namespace gridtools{

    template< enumtype::backend BackendId, enumtype::strategy StrategyType >
    struct backend : public backend_base<BackendId, StrategyType>
    {
        typedef backend_base<BackendId, StrategyType> base_t;

        using base_t::backend_traits_t;
        using base_t::strategy_traits_t;

        static const enumtype::strategy s_strategy_id=base_t::s_strategy_id;
        static const enumtype::backend s_backend_id =base_t::s_backend_id;

    };

} //namespace gridtools
