#pragma once
#include "../../common/generic_metafunctions/variadic_to_vector.hpp"
#include "../../common/generic_metafunctions/accumulate.hpp"
/**
   @file metafunctions used in the cache_storage class
*/

namespace gridtools {

    namespace _impl {

        template < typename Layout, typename Plus, typename Minus, typename Tiles, ushort_t NColors, typename Storage >
        struct compute_meta_storage;

        /**
           @class computing the correct storage_info type for the cache storage
           \tparam Layout memory layout of the cache storage
           \tparam Plus the positive extents in all directions
           \tparam Plus the negative extents in all directions

           The extents and block size are used to compute the dimension of the cache storage, which is
           all we need.
         */
        template < typename Layout,
            typename P1,
            typename P2,
            typename... Plus,
            typename M1,
            typename M2,
            typename... Minus,
            typename T1,
            typename T2,
            typename... Tiles,
            ushort_t NColors,
            typename Storage >
        struct compute_meta_storage< Layout,
            variadic_to_vector< P1, P2, Plus... >,
            variadic_to_vector< M1, M2, Minus... >,
            variadic_to_vector< T1, T2, Tiles... >,
            NColors,
            Storage > {

            typedef meta_storage_cache< Layout,
                P1::value - M1::value + T1::value,
// HACK
#ifndef STRUCTURED_GRIDS
                NColors,
#endif
                P2::value - M2::value + T2::value, // first 2 dimensions are special (the block)
                ((Plus::value - Minus::value) > 0 ? (Tiles::value - Minus::value + Plus::value) : 1)...,
                Storage::field_dimensions,
                1 > type;
        };

        namespace impl {
            template < ushort_t D >
            struct get_layout_map_;

            template <>
            struct get_layout_map_< 2 > {
                typedef layout_map< 1, 0 > type;
            };

            template <>
            struct get_layout_map_< 3 > {
                typedef layout_map< 2, 1, 0 > type;
            };

            template <>
            struct get_layout_map_< 4 > {
                typedef layout_map< 3, 2, 1, 0 > type;
            };

            template <>
            struct get_layout_map_< 5 > {
                typedef layout_map< 4, 3, 2, 1, 0 > type;
            };

            template <>
            struct get_layout_map_< 6 > {
                typedef layout_map< 5, 4, 3, 2, 1, 0 > type;
            };
        }

        template < typename T >
        struct generate_layout_map;

        /**@class automatically generates the layout map for the cache storage. By default
           i and j have the smallest stride. The largest stride is in the field dimension. This reduces bank conflicts.
         */
        template < uint_t... Id >
        struct generate_layout_map< gt_integer_sequence< uint_t, Id... > > {
#ifdef CUDA8
            typedef layout_map< (sizeof...(Id)-Id - 1)... > type;
#else
            typedef typename impl::get_layout_map_< sizeof...(Id) >::type type;
#endif
        };

#ifndef CUDA8
        template < typename Minus, typename Plus, typename Tiles, typename Storage >
        struct compute_size;

        template < typename... Minus, typename... Plus, typename... Tiles, typename Storage >
        struct compute_size< variadic_to_vector< Minus... >,
            variadic_to_vector< Plus... >,
            variadic_to_vector< Tiles... >,
            Storage > {
            static constexpr auto value =
                accumulate(multiplies(), (Plus::value + Tiles::value - Minus::value)...) * Storage::field_dimensions;
        };
#endif

    } // namespace _impl
} // namespace gridtools
