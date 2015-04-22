#pragma once

#include "backend.h"
namespace gridtools {

//traits for backend
template<typename T> struct is_backend : boost::mpl::false_{};

template<enumtype::backend BackendId, enumtype::strategy StrategyType >
struct is_backend<backend<BackendId, StrategyType> > : boost::mpl::true_{};

template<typename T> struct backend_id;

template< enumtype::backend BackendId, enumtype::strategy StrategyType >
struct backend_id< backend<BackendId, StrategyType> > :
    boost::mpl::integral_c<enumtype::backend, BackendId>{};

} // gridtools
