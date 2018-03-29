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
#include "generic_metafunctions/gt_integer_sequence.hpp"
#include "generic_metafunctions/meta.hpp"
#include <vector>
#include <array>

namespace gridtools {
    template < typename T, size_t D >
    std::ostream &operator<<(std::ostream &s, array< T, D > const &a) {
        s << " {  ";
        for (int i = 0; i < D - 1; ++i) {
            s << a[i] << ", ";
        }
        s << a[D - 1] << "  } ";

        return s;
    }

    template < typename T, size_t D >
    std::vector< T > to_vector(array< T, D > const &a) {
        std::vector< T > v(D);
        for (int i = 0; i < D; ++i) {
            v.at(i) = a[i];
        }
        return v;
    }

    namespace impl {
        template < typename Value >
        struct array_initializer {
            template < int Idx >
            struct type {
                type() = delete;

                template < long unsigned int ndims >
                GT_FUNCTION constexpr static Value apply(const std::array< Value, ndims > data) {
                    return data[Idx];
                }

                template < long unsigned int ndims >
                GT_FUNCTION constexpr static Value apply(const gridtools::array< Value, ndims > data) {
                    return data[Idx];
                }
            };
        };
    }

    namespace impl_ {
        // TODO maybe needed for input type checking
        //        template < typename T, typename Enable = void >
        //        struct has_tuple_concept : public std::false_type {};
        //
        //        template < typename T >
        //        struct has_tuple_concept< T,
        //            meta::void_t< std::integral_constant< size_t, tuple_size< typename std::decay< T >::type >::value
        //            > > >
        //            : public std::true_type {};

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

        template < class, class >
        struct convert_to_f;

        template < class NewT, size_t... Is >
        struct convert_to_f< NewT, gt_index_sequence< Is... > > {
            template < typename Container, typename Res = array< NewT, sizeof...(Is) > >
            GT_FUNCTION Res operator()(Container &&a) {
                return {static_cast< NewT >(get< Is >(std::forward< Container >(a)))...};
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

    /**
     * @brief convert tuple-like container to array<NewT,D>, where NewT is explicit or std::common_type.
     * use-cases:
     * a) convert the type of array elements, e.g. convert_to<int>(array<size_t,X>) -> array<int,X>
     * b) convert a tuple or pair to an array e.g. convert_to<size_t>(tuple<size_t,size_t>) -> array<size_t,2>
     */
    template < typename NewT, typename Container >
    GT_FUNCTION auto convert_to(Container &&a) GT_AUTO_RETURN((impl_::convert_to_f< NewT,
        make_gt_index_sequence< tuple_size< typename std::decay< Container >::type >::value > >{}(
        std::forward< Container >(a))));

} // namespace gridtools

template < typename T, typename U, size_t D >
bool same_elements(gridtools::array< T, D > const &a, gridtools::array< U, D > const &b) {
    // shortcut
    if (a.size() != b.size())
        return false;

    // sort and check for equivalence
    gridtools::array< T, D > a0 = a;
    gridtools::array< U, D > b0 = b;
    std::sort(a0.begin(), a0.end());
    std::sort(b0.begin(), b0.end());
    return std::equal(a0.begin(), a0.end(), b0.begin());
}
