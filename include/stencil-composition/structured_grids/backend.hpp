#pragma once

#include "stencil-composition/backend_base.hpp"

namespace gridtools{

    template< enumtype::platform BackendId, enumtype::strategy StrategyType >
    struct backend<BackendId, enumtype::structured, StrategyType> :
            public backend_base<BackendId, enumtype::structured, StrategyType>
    {
        typedef backend_base<BackendId, enumtype::structured, StrategyType> base_t;

        using typename base_t::backend_traits_t;
        using typename base_t::strategy_traits_t;
        using typename base_t::grid_traits_t;

        static const enumtype::strategy s_strategy_id=base_t::s_strategy_id;
        static const enumtype::platform s_backend_id =base_t::s_backend_id;
        static const enumtype::grid_type s_grid_type_id = base_t::s_grid_type_id;
    };

#ifndef CXX11_ENABLED

#ifdef __CUDACC__
    template < ushort_t Index, typename Layout, typename Halo >
    struct is_meta_storage<typename backend<enumtype::Cuda, enumtype::Block>::template storage_info<Index, Layout, Halo > > : boost::mpl::true_{};

    template < ushort_t Index, typename Layout, typename Halo >
    struct is_meta_storage<typename backend<enumtype::Cuda, enumtype::Naive>::template storage_info<Index, Layout, Halo > > : boost::mpl::true_{};
#else
    template < ushort_t Index, typename Layout, typename Halo >
    struct is_meta_storage<typename backend<enumtype::Host, enumtype::Block>::template storage_info<Index, Layout, Halo > > : boost::mpl::true_{};

    template < ushort_t Index, typename Layout, typename Halo >
    struct is_meta_storage<typename backend<enumtype::Host, enumtype::Naive>::template storage_info<Index, Layout, Halo > > : boost::mpl::true_{};
#endif

#endif

} //namespace gridtools
