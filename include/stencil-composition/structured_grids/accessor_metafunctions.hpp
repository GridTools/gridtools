#pragma once

#include "../global_accessor.hpp"
#include "./accessor.hpp"

namespace gridtools {

    template < typename T >
    struct is_accessor : boost::mpl::false_ {};

    template < ushort_t ID, enumtype::intend Intend, typename Extend, ushort_t Number >
    struct is_accessor< accessor< ID, Intend, Extend, Number > > : boost::mpl::true_ {};

    template < ushort_t ID, enumtype::intend Intend >
    struct is_accessor< global_accessor< ID, Intend > > : boost::mpl::true_ {};

#if defined(CXX11_ENABLED) && !defined(__CUDACC__) && !defined(__INTEL_COMPILER)
    template < typename ArgType, typename... Pair >
    struct is_accessor< accessor_mixed< ArgType, Pair... > > : boost::mpl::true_ {};
#endif

    // TODOMEETING accessor_index should be common to all grids
    template < typename Accessor >
    struct accessor_index {
        GRIDTOOLS_STATIC_ASSERT((is_accessor< Accessor >::value), "Internal Error: wrong type");
        typedef typename Accessor::index_type type;
    };

    /**
     * @brief metafunction that given an accesor and a map, it will remap the index of the accessor according
     * to the corresponding entry in ArgsMap
     */
    template < typename Accessor, typename ArgsMap, typename Enable = void >
    struct remap_accessor_type {};

    template < ushort_t ID, enumtype::intend Intend, typename Extend, ushort_t Number, typename ArgsMap >
    struct remap_accessor_type< accessor< ID, Intend, Extend, Number >, ArgsMap > {
        typedef accessor< ID, Intend, Extend, Number > accessor_t;
        GRIDTOOLS_STATIC_ASSERT((boost::mpl::size< ArgsMap >::value > 0), "Internal Error: wrong size");
        // check that the key type is an int (otherwise the later has_key would never find the key)
        GRIDTOOLS_STATIC_ASSERT(
            (boost::is_same<
                typename boost::mpl::first< typename boost::mpl::front< ArgsMap >::type >::type::value_type,
                int >::value),
            "Internal Error");

        typedef typename boost::mpl::integral_c< int, (int)ID > index_type_t;

        GRIDTOOLS_STATIC_ASSERT((boost::mpl::has_key< ArgsMap, index_type_t >::value), "Internal Error");

        typedef accessor< boost::mpl::at< ArgsMap, index_type_t >::type::value, Intend, Extend, Number > type;
    };

    template < ushort_t ID, enumtype::intend Intend, typename ArgsMap >
    struct remap_accessor_type< global_accessor< ID, Intend >, ArgsMap > {
        typedef global_accessor< ID, Intend > accessor_t;
        GRIDTOOLS_STATIC_ASSERT((boost::mpl::size< ArgsMap >::value > 0), "Internal Error: wrong size");
        // check that the key type is an int (otherwise the later has_key would never find the key)
        GRIDTOOLS_STATIC_ASSERT(
            (boost::is_same<
                typename boost::mpl::first< typename boost::mpl::front< ArgsMap >::type >::type::value_type,
                int >::value),
            "Internal Error");

        typedef typename boost::mpl::integral_c< int, (int)ID > index_type_t;

        GRIDTOOLS_STATIC_ASSERT((boost::mpl::has_key< ArgsMap, index_type_t >::value), "Internal Error");

        typedef global_accessor< boost::mpl::at< ArgsMap, index_type_t >::type::value, Intend > type;
    };

#ifdef CXX11_ENABLED
    template < typename ArgsMap, template < typename... > class Expression, typename... Arguments >
    struct remap_accessor_type< Expression< Arguments... >,
        ArgsMap,
        typename boost::enable_if< typename is_expr< Expression< Arguments... > >::type, void >::type > {
        // Expression is an expression of accessors (e.g. expr_sum<T1, T2>,
        // where T1 and T2 are two accessors).
        // Here we traverse the expression AST down to the leaves, and we assert if
        // the leaves are not accessor types.

        // recursively remapping the template arguments,
        // until the specialization above stops the recursion
        typedef Expression< typename remap_accessor_type< Arguments, ArgsMap >::type... > type;
    };

    // Workaround needed to prevent nvcc to instantiate the struct in enable_ifs
    template < typename ArgsMap, template < typename... > class Expression, typename... Arguments >
    struct remap_accessor_type< Expression< Arguments... >,
        ArgsMap,
        typename boost::disable_if< typename is_expr< Expression< Arguments... > >::type, void >::type > {
        // Workaround needed to prevent nvcc to instantiate the struct in enable_ifs
        typedef boost::mpl::void_ type;
    };

    template < typename ArgsMap >
    struct remap_accessor_type< float_type, ArgsMap > {
        // when a leaf is a float don't do anything
        typedef float_type type;
    };

    template < typename ArgsMap, template < typename Acc, int N > class Expression, typename Accessor, int Number >
    struct remap_accessor_type< Expression< Accessor, Number >, ArgsMap > {
        // Specialization done to catch also the "pow" expression, for which a template argument is an
        // integer (the exponent)
        typedef Expression< typename remap_accessor_type< Accessor, ArgsMap >::type, Number > type;
    };

#endif

    template < typename Accessor >
    struct is_accessor_readonly : boost::mpl::false_ {};

#if defined(CXX11_ENABLED) && !defined(CUDA_CXX11_BUG_1) && !defined(__INTEL_COMPILER)
    template < typename Accessor, typename... Pair >
    struct is_accessor_readonly< accessor_mixed< Accessor, Pair... > > : is_accessor_readonly< Accessor > {};
#endif

    template < ushort_t ID, typename Extend, ushort_t Number >
    struct is_accessor_readonly< accessor< ID, enumtype::in, Extend, Number > > : boost::mpl::true_ {};

    template < ushort_t ID, typename Extend, ushort_t Number >
    struct is_accessor_readonly< accessor< ID, enumtype::inout, Extend, Number > > : boost::mpl::false_ {};

    template < ushort_t ID >
    struct is_accessor_readonly< global_accessor< ID, enumtype::in > > : boost::mpl::true_ {};

    template < ushort_t ID >
    struct is_accessor_readonly< global_accessor< ID, enumtype::inout > > : boost::mpl::true_ {};

    /* Is written is actually "can be written", since it checks if not read olnly.
       TODO: metafunction convention not completely respected */
    template < typename Accessor >
    struct is_accessor_written {
        static const bool value = !is_accessor_readonly< Accessor >::value;
        typedef typename boost::mpl::not_< typename is_accessor_readonly< Accessor >::type >::type type;
    };

} // namespace gridtools
