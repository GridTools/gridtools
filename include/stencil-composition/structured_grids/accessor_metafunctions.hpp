/*
  GridTools Libraries

  Copyright (c) 2016, GridTools Consortium
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  1. Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

  For information: http://eth-cscs.github.io/gridtools/
*/
#pragma once

#include "../global_accessor.hpp"
#include "./accessor.hpp"
#include "./accessor_mixed.hpp"
#ifdef CXX11_ENABLED
#include "../expressions/expressions.hpp"
#endif

namespace gridtools {

    template < typename T >
    struct is_accessor : boost::mpl::false_ {};

    template < ushort_t ID, enumtype::intend Intend, typename Extend, ushort_t Number >
    struct is_accessor< accessor< ID, Intend, Extend, Number > > : boost::mpl::true_ {};

    template < ushort_t ID, enumtype::intend Intend, typename Extend, ushort_t Number >
    struct is_accessor< accessor_base< ID, Intend, Extend, Number > > : boost::mpl::true_ {};

    template < typename T >
    struct is_accessor< const T > : is_accessor< T > {};

    template < ushort_t ID, enumtype::intend Intend >
    struct is_accessor< global_accessor< ID, Intend > > : boost::mpl::true_ {};

#ifdef CUDA8
    template < typename ArgType >
    struct is_accessor_mixed;

    template < typename... Types >
    struct is_accessor_mixed< accessor_mixed< Types... > > : boost::mpl::true_ {};

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

#ifdef CUDA8
    template < typename Accessor, typename ArgsMap, typename... Pairs >
    struct remap_accessor_type< accessor_mixed< Accessor, Pairs... >, ArgsMap > {

        typedef typename remap_accessor_type< Accessor, ArgsMap >::index_type_t index_type_t;

        typedef accessor_mixed< typename remap_accessor_type< Accessor, ArgsMap >::type, Pairs... > type;
    };
#endif

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

#ifdef CUDA8
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
