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

#include <cmath>

#include <boost/mpl/and.hpp>
#include <boost/type_traits.hpp>

#include "alignment.hpp"
#include "halo.hpp"
#include "../../common/array.hpp"
#include "../../common/variadic_pack_metafunctions.hpp"
#include "../../common/layout_map.hpp"
#include "../../common/gt_math.hpp"

namespace gridtools {

    /* forward declaration */
    template < typename T >
    struct is_alignment;

    /*
     * @brief struct used to extend a given number of dimensions with a given halo
     * @tparam HaloVal halo size
     * @tparam LayoutArg layout map entry
     */
    template < unsigned HaloVal, int LayoutArg >
    struct extend_by_halo {
        template < typename Dim >
        static constexpr unsigned extend(Dim d) {
            static_assert(boost::is_integral< Dim >::value, "Dimensions has to be integral type.");
            return error_or_return((d > 0),
                ((LayoutArg == -1) ? 1 : d + 2 * HaloVal),
                "Tried to instantiate storage info with zero or negative dimensions");
        }
    };

    /*
     * @brief function used to provide an aligned dimension
     * @tparam Alignment alignment information
     * @tparam Length Layout map length
     * @tparam LayoutArg layout map entry
     * @return return aligned dimension if it should be aligned, otherwise return as is.
     */
    template < typename Alignment, unsigned Length, int LayoutArg >
    constexpr unsigned align_dimensions(unsigned dimension) {
        static_assert(is_alignment< Alignment >::value, "Passed type is no alignment type");
        return ((Alignment::value > 1) && (LayoutArg == Length - 1))
                   ? gt_ceil((float)dimension / (float)Alignment::value) * Alignment::value
                   : dimension;
    }

    /*
     * @brief struct used to compute the initial offset. The initial offset
     * has to be added if we use alignment in combination with a halo
     * in the aligned dimension. We want to have the first non halo point
     * aligned. Therefore we have to introduce an initial offset.
     * @tparam Layout layout map
     * @tparam Alignment alignment information
     * @tparam Halo halo information
     */
    template < typename Layout, typename Alignment, typename Halo >
    struct get_initial_offset;

    template < int... LayoutArgs, typename Alignment, unsigned... HaloVals >
    struct get_initial_offset< layout_map< LayoutArgs... >, Alignment, halo< HaloVals... > > {
        static_assert(is_alignment< Alignment >::value, "Passed type is no alignment type");

        // this function returns the halo of the first stride dimension
        // e.g., layout_map<1,2,-1,0> with halo<3,4,0,1> would return 4
        // because the second dimension is the one first stride dimension
        // and the halo there is 4.
        constexpr static unsigned get_first_stride_dim_halo() {
            return get_value_from_pack(get_index_of_element_in_pack(0,
                                           static_cast< int >(layout_map< LayoutArgs... >::unmasked_length - 1),
                                           static_cast< int >(LayoutArgs)...),
                static_cast< unsigned >(HaloVals)...);
        }

        // if the alignment is >1 we have to calculate an initial offset.
        // e.g., alignment<32>, layout_map<0,1,2>, halo<0,2,2>. the function
        // gt_ceil(2/32) * 32 - 2 = 1 * 32 - 2 = 30. This means the initial offset
        // is 30, followed by 2 halo points, followed by an aligned non-halo point.
        constexpr static unsigned compute() {
            return ((Alignment::value > 1u) && (get_first_stride_dim_halo() > 0))
                       ? static_cast< unsigned >(
                             gt_ceil((float)get_first_stride_dim_halo() / (float)Alignment::value) * Alignment::value -
                             get_first_stride_dim_halo())
                       : 0;
        }
    };

    /*
     * @brief helper struct used to compute the strides in a constexpr manner
     * @tparam T the layout map type
     */
    template < typename T >
    struct get_strides_aux;

    template < int... LayoutArgs >
    struct get_strides_aux< layout_map< LayoutArgs... > > {
        typedef layout_map< LayoutArgs... > layout_map_t;

        template < int N, typename... Dims >
        static constexpr
            typename boost::enable_if_c< (N != -1 && N != layout_map_t::unmasked_length - 1), unsigned >::type
                get_stride(Dims... d) {
            return (get_value_from_pack(get_index_of_element_in_pack(0, N + 1, LayoutArgs...), d...)) *
                   get_stride< N + 1 >(d...);
        }

        template < int N, typename... Dims >
        static constexpr typename boost::enable_if_c< (N == -1), unsigned >::type get_stride(Dims... d) {
            return 0;
        }

        template < int N, typename... Dims >
        static constexpr
            typename boost::enable_if_c< (N != -1 && N == layout_map_t::unmasked_length - 1), unsigned >::type
                get_stride(Dims... d) {
            return 1;
        }
    };

    /*
     * @brief struct used to compute the strides given the dimensions (e.g., 128x128x80)
     * @tparam Layout layout map
     */
    template < typename Layout >
    struct get_strides;

    template < int... LayoutArgs >
    struct get_strides< layout_map< LayoutArgs... > > {
        template < typename... Dims >
        static constexpr array< unsigned, sizeof...(LayoutArgs) > get_stride_array(Dims... d) {
            static_assert(boost::mpl::and_< boost::mpl::bool_< (sizeof...(Dims) > 0) >,
                              typename is_all_integral< Dims... >::type >::value,
                "Dimensions have to be integral types.");
            typedef layout_map< LayoutArgs... > Layout;
            return (array< unsigned, Layout::masked_length >){
                get_strides_aux< Layout >::template get_stride< LayoutArgs >(d...)...};
        }
    };
}
