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

/**
 * @brief metafunction that given an arg and a map, it will remap the index of the arg according
 * to the corresponding entry in ArgsMap
 */
template<typename Accessor, typename ArgsMap>
struct remap_accessor_type{};

template < ushort_t ID, typename Range, ushort_t Number, typename ArgsMap>
struct remap_accessor_type<accessor<ID, Range, Number>, ArgsMap >
{
    typedef accessor<ID, Range, Number> accessor_t;
    BOOST_STATIC_ASSERT((boost::mpl::size<ArgsMap>::value>0));
    //check that the key type is an int (otherwise the later has_key would never find the key)
    BOOST_STATIC_ASSERT((boost::is_same<
        typename boost::mpl::first<typename boost::mpl::front<ArgsMap>::type>::type::value_type,
        int
    >::value));

    typedef typename boost::mpl::integral_c<int, (int)ID> index_type_t;

    BOOST_STATIC_ASSERT((boost::mpl::has_key<ArgsMap, index_type_t>::value));

    typedef accessor<
        boost::mpl::at<ArgsMap, index_type_t >::type::value,
        Range,
        Number
    > type;
};

} //namespace gridtools
