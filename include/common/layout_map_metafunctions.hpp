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
#include "generic_metafunctions/gt_integer_sequence.hpp"
#include "layout_map.hpp"
#include "selector.hpp"
#include "generic_metafunctions/replace.hpp"
#include "generic_metafunctions/sequence_unpacker.hpp"

#ifdef CXX11_ENABLED
namespace gridtools {

    /*
     * metafunction to filter out some of the dimensions of a layout map, determined by the DimSelector
     * Example of use: filter_layout<layout_map<0,1,2,3>, selector<1,1,-1,1 > == layout_map<0,1,-1,2>
     */
    template < typename Layout, typename DimSelector >
    struct filter_layout {
        GRIDTOOLS_STATIC_ASSERT((is_selector< DimSelector >::value), "Error");
        GRIDTOOLS_STATIC_ASSERT((is_layout_map< Layout >::value), "Error: need a layout map type");
        GRIDTOOLS_STATIC_ASSERT(
            (DimSelector::length >= Layout::length), "Error: need to specifiy at least 4 dimensions");

        template < uint_t NumNullDims, typename Seq_ >
        struct data_ {
            typedef Seq_ seq_t;
            static constexpr const uint_t num_null_dims = NumNullDims;
        };

        template < typename Data, typename Index >
        struct insert_index_at_pos {

            template < typename Data_, typename Position >
            struct insert_a_null {
                typedef data_< Data_::num_null_dims + 1,
                    typename replace< typename Data_::seq_t, Position, static_int< -1 > >::type > type;
            };

            template < typename Data_, typename Position >
            struct insert_a_pos_index {
                typedef data_<
                    Data_::num_null_dims,
                    typename replace< typename Data_::seq_t,
                        Position,
                        static_int< Layout::template at_< Position::value >::value - Data_::num_null_dims > >::type >
                    type;
            };
            typedef static_int< Layout::template pos_< Index::value >::value > position_t;
            typedef typename boost::mpl::if_c< (DimSelector::template get_elem< position_t::value >::value == -1),
                typename insert_a_null< Data, position_t >::type,
                typename insert_a_pos_index< Data, position_t >::type >::type type;
        };

        typedef typename boost::mpl::fold< boost::mpl::range_c< int, 0, Layout::length >,
            boost::mpl::vector0<>,
            boost::mpl::push_back< boost::mpl::_1, static_int< 0 > > >::type initial_vector;

        typedef data_< 0, initial_vector > initial_data;
        typedef typename boost::mpl::fold< boost::mpl::range_c< int, 0, Layout::length >,
            initial_data,
            insert_index_at_pos< boost::mpl::_1, boost::mpl::_2 > >::type new_layout_indices_data_t;

        typedef typename new_layout_indices_data_t::seq_t new_layout_indices_t;

        template < typename T >
        struct indices_to_layout;

        template < typename... Args >
        struct indices_to_layout< variadic_typedef< Args... > > {
            using type = layout_map< Args::value... >;
        };

        typedef typename indices_to_layout< typename sequence_unpacker< new_layout_indices_t >::type >::type type;
    };

    namespace impl {
        template < int_t Val, short_t NExtraDim >
        struct inc_ {
            static const int_t value = Val == -1 ? -1 : Val + NExtraDim;
        };
    }

    template < typename LayoutMap, ushort_t NExtraDim >
    struct extend_layout_map;

    /*
     * metafunction to extend a layout_map with certain number of dimensions.
     * Example of use: extend_layout_map< layout_map<0, 1, 3, 2>, 3> == layout_map<3, 4, 6, 5, 0, 1, 2>
     */
    template < ushort_t NExtraDim, int_t... Args >
    struct extend_layout_map< layout_map< Args... >, NExtraDim > {

        template < typename T, int_t... InitialInts >
        struct build_ext_layout;

        // build an extended layout
        template < int_t... Indices, int_t... InitialIndices >
        struct build_ext_layout< gt_integer_sequence< int_t, Indices... >, InitialIndices... > {
            typedef layout_map< InitialIndices..., Indices... > type;
        };

        using seq = typename make_gt_integer_sequence< int_t, NExtraDim >::type;

        typedef typename build_ext_layout< seq, impl::inc_< Args, NExtraDim >::value... >::type type;
    };
}
#endif
