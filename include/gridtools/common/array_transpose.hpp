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
        struct get_inner_type {
            using type = typename std::decay< decltype(get< 0 >(get< 0 >(std::declval< T >()))) >::type;
        };

        template < typename T >
        struct get_inner_dim : tuple_size< typename std::decay< decltype(get< 0 >(std::declval< T >())) >::type > {};

        template < typename T >
        struct get_outer_dim : tuple_size< T > {};

        template < typename T >
        struct new_inner_type {
            using type = array< typename get_inner_type< T >::type, get_outer_dim< T >::value >;
        };

        template < typename T >
        struct meta_transpose {
            using type = array< typename new_inner_type< T >::type, get_inner_dim< T >::value >;
        };

        template < class Res, class Indices >
        struct array_transform_f;
        template < class Res, size_t... Is >
        struct array_transform_f< Res, gt_index_sequence< Is... > > {
            template < class Fun, class Src >
            GT_FUNCTION Res operator()(Fun &&fun, Src &&src) const {
                return {std::forward< Fun >(fun)(get< Is >(std::forward< Src >(src)))...};
            }
        };

        template < size_t I >
        struct get_f {
            template < class Src >
            GT_FUNCTION auto operator()(Src &&src) const GT_AUTO_RETURN(get< I >(std::forward< Src >(src)));
        };

        template < size_t I,
            class T,
            class Decayed = typename std::decay< T >::type,
            class Res = typename new_inner_type< Decayed >::type >
        GT_FUNCTION Res get_new_inner(T &&obj) {
            return array_transform_f< Res, make_gt_index_sequence< get_outer_dim< Decayed >::value > >{}(
                get_f< I >{}, std::forward< T >(obj));
        }

        template < class >
        struct transpose_f;

        template < size_t... Is >
        struct transpose_f< gt_index_sequence< Is... > > {
            template < class T, class Res = typename meta_transpose< typename std::decay< T >::type >::type >
            GT_FUNCTION Res operator()(T &&obj) const {
                return {get_new_inner< Is >(std::forward< T >(obj))...};
            }
        };
    }

    /**
     * @brief transposes array<array<T,InnerDim>,OuterDim> into array<array<T,OuterDim>,InnerDim>
     */
    template < typename Container >
    GT_FUNCTION auto transpose(Container &&a) GT_AUTO_RETURN(impl_::transpose_f<
        make_gt_index_sequence< impl_::get_inner_dim< typename std::decay< Container >::type >::value > >()(
        std::forward< Container >(a)));

} // namespace gridtools
