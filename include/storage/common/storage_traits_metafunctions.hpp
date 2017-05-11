/*
  GridTools Libraries

  Copyright (c) 2017, GridTools Consortium
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

#include "../../common/layout_map.hpp"
#include "../../common/selector.hpp"

namespace gridtools {

    /* Layout map extender, takes a given layout and extends it by n dimensions (ascending and descending version) */
    template < unsigned Dim, unsigned Current, typename Layout >
    struct layout_map_ext_asc;

    template < unsigned Dim, unsigned Current, int... Dims >
    struct layout_map_ext_asc< Dim, Current, layout_map< Dims... > >
        : layout_map_ext_asc< Dim - 1, Current + 1, layout_map< Dims..., Current > > {};

    template < unsigned Current, int... Dims >
    struct layout_map_ext_asc< 0, Current, layout_map< Dims... > > {
        typedef layout_map< Dims... > type;
    };

    template < unsigned Ext, typename Layout >
    struct layout_map_ext_dsc;

    template < unsigned Ext, int... Dims >
    struct layout_map_ext_dsc< Ext, layout_map< Dims... > >
        : layout_map_ext_dsc< Ext - 1, layout_map< Dims..., Ext - 1 > > {};

    template < int... Dims >
    struct layout_map_ext_dsc< 0, layout_map< Dims... > > {
        typedef layout_map< Dims... > type;
    };

    /* get a standard layout_map (either 3 or n-dimensional and ascending or descending) */
    template < unsigned Dim, bool Asc >
    struct get_layout;

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

    // get a multidimensional layout in ascending order (e.g., host backend)
    template < unsigned Dim >
    struct get_layout< Dim, true > {
        static_assert(Dim > 0, "Zero dimensional layout makes no sense.");
        typedef typename layout_map_ext_asc< Dim - 3, 0, layout_map< Dim - 3, Dim - 2, Dim - 1 > >::type type;
    };

    // get a multidimensional layout in descending order (e.g., gpu backend)
    template < unsigned Dim >
    struct get_layout< Dim, false > {
        static_assert(Dim > 0, "Zero dimensional layout makes no sense.");
        typedef typename layout_map_ext_dsc< Dim - 1, layout_map< Dim - 1 > >::type type;
    };

    /* special layout construction mechanisms */
    template < int Dim, bool Bit >
    struct select_dimension {
        const static int value = (Bit) ? Dim : -1;
    };

    template < int Dim, bool Bit >
    struct select_non_dimension {
        const static int value = (!Bit) ? Dim : -1;
    };

    template < typename Vec, typename T >
    struct fix_values;

    template < typename ValueVec, int... D >
    struct fix_values< ValueVec, layout_map< D... > > {
        typedef layout_map< boost::mpl::if_< boost::is_same< boost::mpl::int_< -1 >, boost::mpl::int_< D > >,
            boost::mpl::int_< -1 >,
            boost::mpl::int_< (int)(D - (int)boost::mpl::count_if< ValueVec,
                                            boost::mpl::less< boost::mpl::_, boost::mpl::int_< D > > >::type::
                                            value) > >::type::value... >
            type;
    };

    template < typename T, typename Selector >
    struct get_special_layout;

    template < int... Dims, bool... Bitmask >
    struct get_special_layout< layout_map< Dims... >, selector< Bitmask... > > {
        // <1,1,0,0,1,1>
        typedef typename get_mpl_vector< boost::mpl::vector<>, Bitmask... >::type bitmask_vec;
        static_assert(
            boost::mpl::count_if< bitmask_vec, boost::is_same< boost::mpl::int_< -1 >, boost::mpl::_1 > >::value <
                sizeof...(Dims),
            "Masking out all dimensions makes no sense.");
        // <1,2,3,4,5,0>
        typedef typename get_mpl_vector< boost::mpl::vector<>, Dims... >::type dims_vec;
        // <3,4>
        typedef typename boost::mpl::remove<
            typename get_mpl_vector< boost::mpl::vector<>, select_non_dimension< Dims, Bitmask >::value... >::type,
            boost::mpl::int_< -1 > >::type masked_vec;
        typedef typename fix_values< masked_vec, layout_map< select_dimension< Dims, Bitmask >::value... > >::type type;
    };
}
