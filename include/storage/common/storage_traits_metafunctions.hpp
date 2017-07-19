/*
  GridTools Libraries

  Copyright (c) 2017, ETH Zurich and MeteoSwiss
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

#include <boost/mpl/count_if.hpp>
#include <boost/mpl/int.hpp>
#include <boost/mpl/push_back.hpp>
#include <boost/mpl/remove.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/type_traits.hpp>

#include "../../common/gt_assert.hpp"
#include "../../common/layout_map.hpp"
#include "../../common/selector.hpp"
#include "../../common/generic_metafunctions/variadic_to_vector.hpp"

namespace gridtools {

    /* Layout map extender, takes a given layout and extends it by n dimensions (ascending and descending version) */
    template < uint_t Dim, uint_t Current, typename Layout >
    struct layout_map_ext_asc;

    template < uint_t Dim, uint_t Current, int... Dims >
    struct layout_map_ext_asc< Dim, Current, layout_map< Dims... > >
        : layout_map_ext_asc< Dim - 1, Current + 1, layout_map< Dims..., Current > > {};

    template < uint_t Current, int... Dims >
    struct layout_map_ext_asc< 0, Current, layout_map< Dims... > > {
        typedef layout_map< Dims... > type;
    };

    template < uint_t Ext, typename Layout >
    struct layout_map_ext_dsc;

    template < uint_t Ext, int... Dims >
    struct layout_map_ext_dsc< Ext, layout_map< Dims... > >
        : layout_map_ext_dsc< Ext - 1, layout_map< Dims..., Ext - 1 > > {};

    template < int... Dims >
    struct layout_map_ext_dsc< 0, layout_map< Dims... > > {
        typedef layout_map< Dims... > type;
    };

    /* get a standard layout_map (n-dimensional and ascending or descending) */
    template < uint_t Dim, bool Asc >
    struct get_layout;

    // get a multidimensional layout in ascending order (e.g., host backend)
    /**
     * @brief metafunction used to retrieve a layout_map with n-dimensions
     * that can be used in combination with the host backend (k-first order).
     * E.g., get_layout< 5, true > will return following type: layout_map< 2, 3, 4, 0, 1 >.
     * This means the k-dimension (value: 4) is coalesced in memory, followed
     * by the j-dimension (value: 3), followed by the i-dimension (value: 2), followed
     * by the fifth dimension (value: 1), etc. The reason for having k as innermost
     * is because of the gridtools execution model. The CPU backend will give best
     * performance (in most cases) when using the provided layout.
     */
    template < uint_t Dim >
    struct get_layout< Dim, true > {
        GRIDTOOLS_STATIC_ASSERT(Dim > 0, GT_INTERNAL_ERROR_MSG("Zero dimensional layout makes no sense."));
        typedef typename layout_map_ext_asc< Dim - 3, 0, layout_map< Dim - 3, Dim - 2, Dim - 1 > >::type type;
    };

    // get a multidimensional layout in descending order (e.g., gpu backend)
    /**
     * @brief metafunction used to retrieve a layout_map with n-dimensions
     * that can be used in combination with the GPU backend (i-first order).
     * E.g., get_layout< 5, false > will return following type: layout_map< 4, 3, 2, 1, 0 >.
     * This means the i-dimension (value: 4) is coalesced in memory, followed
     * by the j-dimension (value: 3), followed by the k-dimension (value: 2), followed
     * by the fourth dimension (value: 1), etc. The reason for having i as innermost
     * is because of the gridtools execution model. The GPU backend will give best
     * performance (in most cases) when using the provided layout.
     */
    template < uint_t Dim >
    struct get_layout< Dim, false > {
        GRIDTOOLS_STATIC_ASSERT(Dim > 0, GT_INTERNAL_ERROR_MSG("Zero dimensional layout makes no sense."));
        typedef typename layout_map_ext_dsc< Dim - 1, layout_map< Dim - 1 > >::type type;
    };

    /* specializations up to 3-dimensional for both i-first and k-first layouts */
    template <>
    struct get_layout< 1, true > {
        typedef layout_map< 0 > type;
    };

    template <>
    struct get_layout< 1, false > {
        typedef layout_map< 0 > type;
    };

    template <>
    struct get_layout< 2, true > {
        typedef layout_map< 0, 1 > type;
    };

    template <>
    struct get_layout< 2, false > {
        typedef layout_map< 1, 0 > type;
    };

    template <>
    struct get_layout< 3, true > {
        typedef layout_map< 0, 1, 2 > type;
    };

    template <>
    struct get_layout< 3, false > {
        typedef layout_map< 2, 1, 0 > type;
    };

    /* special layout construction mechanisms */

    /**
     * @brief simple mechanism that is used when iterating through
     * a list of integers and a given selector. If the selector is 1
     * the current integer is returned otherwise -1 is returned.
     * @tparam Dim integer
     * @tparam Bit either true or false
     */
    template < int Dim, bool Bit >
    struct select_dimension {
        const static int value = (Bit) ? Dim : -1;
    };

    /**
     * @brief negation of select_dimension
     * @tparam Dim integer
     * @tparam Bit either true or false
     */
    template < int Dim, bool Bit >
    struct select_non_dimension {
        const static int value = (!Bit) ? Dim : -1;
    };

    /**
     * @brief helper metafunction. when we create a layout_map with n-dimensions we
     * have to fix the values when dimensions are masked.
     * E.g., layout_map<3,2,-1,0> has to be fixed to layout_map<2,1,-1,0>
     * @tparam Vec selector
     * @tparam T the layout_map type
     */
    template < typename Vec, typename T >
    struct fix_values;

    template < typename ValueVec, int... D >
    struct fix_values< ValueVec, layout_map< D... > > {
        typedef layout_map< boost::mpl::if_< boost::is_same< boost::mpl::int_< -1 >, boost::mpl::int_< D > >,
            boost::mpl::int_< -1 >,
            boost::mpl::int_< (int)(
                D - (int)boost::mpl::count_if< ValueVec,
                        boost::mpl::less< boost::mpl::_, boost::mpl::int_< D > > >::type::value) > >::type::value... >
            type;
    };

    /**
     * @brief metafunction used to retrieve special layout_map.
     * Special layout_map are layout_maps with masked dimensions.
     * @tparam T the layout_map type
     * @tparam Selector the selector type
     */
    template < typename T, typename Selector >
    struct get_special_layout;

    template < int... Dims, bool... Bitmask >
    struct get_special_layout< layout_map< Dims... >, selector< Bitmask... > > {
        // <1,1,0,0,1,1>
        typedef typename variadic_to_vector< boost::mpl::int_< Bitmask >... >::type bitmask_vec;
        GRIDTOOLS_STATIC_ASSERT(
            (boost::mpl::count_if< bitmask_vec, boost::is_same< boost::mpl::int_< -1 >, boost::mpl::_1 > >::value <
                sizeof...(Dims)),
            GT_INTERNAL_ERROR_MSG("Masking out all dimensions makes no sense."));
        // <1,2,3,4,5,0>
        typedef typename variadic_to_vector< boost::mpl::int_< Dims >... >::type dims_vec;
        // <3,4>
        typedef typename boost::mpl::remove<
            typename variadic_to_vector< boost::mpl::int_< select_non_dimension< Dims, Bitmask >::value >... >::type,
            boost::mpl::int_< -1 > >::type masked_vec;
        typedef typename fix_values< masked_vec, layout_map< select_dimension< Dims, Bitmask >::value... > >::type type;
    };
}
