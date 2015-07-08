#pragma once

#include "accessor.h"

namespace gridtools{

template<typename T>
struct is_accessor : boost::mpl::false_{};

template < ushort_t ID, typename Range, ushort_t Number>
struct is_accessor<accessor<ID, Range, Number> > : boost::mpl::true_{};

#if defined( CXX11_ENABLED) && !defined(__CUDACC__)
template <typename ArgType, typename ... Pair>
struct is_accessor<accessor_mixed<ArgType, Pair ... > > : boost::mpl::true_{};
#endif

template<typename T>
struct accessor_index;

template < ushort_t ID, typename Range, ushort_t Number>
struct accessor_index<accessor<ID, Range, Number> >
{
    typedef boost::mpl::integral_c<ushort_t, ID> type;
    BOOST_STATIC_CONSTANT(ushort_t, value=(type::value));
};

/**
 * @brief metafunction that given an accesor and a map, it will remap the index of the accessor according
 * to the corresponding entry in ArgsMap
 */
template<typename Accessor, typename ArgsMap>
struct remap_accessor_type{};

template < ushort_t ID, typename Range, ushort_t Number, typename ArgsMap>
struct remap_accessor_type<accessor<ID, Range, Number>, ArgsMap >
{
    typedef accessor<ID, Range, Number> accessor_t;
    GRIDTOOLS_STATIC_ASSERT((boost::mpl::size<ArgsMap>::value>0), "Internal Error: wrong size")
    //check that the key type is an int (otherwise the later has_key would never find the key)
    GRIDTOOLS_STATIC_ASSERT((boost::is_same<
        typename boost::mpl::first<typename boost::mpl::front<ArgsMap>::type>::type::value_type,
        int
    >::value), "Internal Error")

    typedef typename boost::mpl::integral_c<int, (int)ID> index_type_t;

    GRIDTOOLS_STATIC_ASSERT((boost::mpl::has_key<ArgsMap, index_type_t>::value), "Internal Error")

    typedef accessor<
        boost::mpl::at<ArgsMap, index_type_t >::type::value,
        Range,
        Number
    > type;
};

#ifdef CX11_ENABLED
    template < typename ArgsMap, template<typename ... > class Expression, typename ... Arguments >
    struct remap_accessor_type<Expression<Arguments ... >, ArgsMap >
    {
        //Expression is an expression of accessors (e.g. expr_sum<T1, T2>,
        //where T1 and T2 are two accessors).
        //Here we traverse the expression AST down to the leaves, and we assert if
        //the leaves are not accessor types.

        //recursively remapping the template arguments,
        //until the specialization above stops the recursion
        typedef Expression<remap_accessor_type<Arguments, ArgsMap> ...> type;
    };
#endif
} //namespace gridtools
