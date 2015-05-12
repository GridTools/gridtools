#pragma once

#include "accessor.h"

namespace gridtools{

template<typename T>
struct is_accessor : boost::mpl::false_{};

template < ushort_t ID, typename Range, ushort_t Number>
struct is_accessor<accessor<ID, Range, Number> > : boost::mpl::true_{};

template<typename T>
struct accessor_index;

template < ushort_t ID, typename Range, ushort_t Number>
struct accessor_index<accessor<ID, Range, Number> >
{
    typedef boost::mpl::integral_c<ushort_t, ID> type;
    BOOST_STATIC_CONSTANT(ushort_t, value=(type::value));
};

} //namespace gridtools
