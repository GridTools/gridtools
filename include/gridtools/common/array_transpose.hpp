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
#include "array.hpp"
#include "common/generic_metafunctions/gt_integer_sequence.hpp"

namespace gridtools {
    namespace impl_ {

        template < typename T >
        using get_inner_type = typename std::decay< decltype(get< 0 >(get< 0 >(T{}))) >::type;

        template < typename T >
        GT_FUNCTION constexpr size_t inner_dim(const T &a) {
            return tuple_size< typename std::decay< decltype(get< 0 >(a)) >::type >::value;
        }

        template < size_t InnerIndex, typename Container, size_t... OuterIndices >
        GT_FUNCTION array< get_inner_type< Container >, sizeof...(OuterIndices) > make_inner_array(
            const Container &a, gt_index_sequence< OuterIndices... >) {
            return {{get< InnerIndex >(get< OuterIndices >(a))...}};
        }

        template < typename Container, size_t... InnerIndices >
        GT_FUNCTION auto transpose_impl(const Container &a, gt_index_sequence< InnerIndices... >) GT_AUTO_RETURN(
            (array< array< get_inner_type< Container >, tuple_size< Container >::value >, sizeof...(InnerIndices) >{
                make_inner_array< InnerIndices >(a, make_gt_index_sequence< tuple_size< Container >::value >())...}));
    }

    /**
     * @brief transposes array<array<T,InnerDim>,OuterDim> into array<array<T,OuterDim>,InnerDim>
     */
    template < typename Container >
    GT_FUNCTION auto transpose(const Container &a)
        GT_AUTO_RETURN((impl_::transpose_impl(a, make_gt_index_sequence< impl_::inner_dim(Container{}) >())));

} // namespace gridtools
