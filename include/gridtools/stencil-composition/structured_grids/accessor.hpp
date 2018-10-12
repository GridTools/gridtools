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

#include <type_traits>

#include "../../common/defs.hpp"
#include "../../common/host_device.hpp"
#include "../accessor_base.hpp"
#include "extent.hpp"
/**
   @file

   @brief File containing the definition of the regular accessor used
   to address the storage (at offsets) from whithin the functors.
   This accessor is a proxy for a storage class, i.e. it is a light
   object used in place of the storage when defining the high level
   computations, and it will be bound later on with a specific
   instantiation of a storage class.

   An accessor can be instantiated directly in the Do
   method, or it might be a constant expression instantiated outside
   the functor scope and with static duration.
*/

namespace gridtools {

    /**
       @brief the definition of accessor visible to the user

       \tparam ID the integer unic ID of the field placeholder

       \tparam Extent the extent of i/j indices spanned by the
               placeholder, in the form of <i_minus, i_plus, j_minus,
               j_plus>.  The values are relative to the current
               position. See e.g. horizontal_diffusion::out_function
               as a usage example.

       \tparam Number the number of dimensions accessed by the
               field. Notice that we don't distinguish at this level what we
               call "space dimensions" from the "field dimensions". Both of
               them are accessed using the same interface. whether they are
               field dimensions or space dimension will be decided at the
               moment of the storage instantiation (in the main function)
     */

    //    namespace _impl {
    //        template <ushort_t I>
    //        struct get_dimension_value_f {
    //            template <ushort_t J>
    //            GT_FUNCTION constexpr int_t operator()(dimension<J> src) const {
    //                return 0;
    //            }
    //            GT_FUNCTION constexpr int_t operator()(dimension<I> src) const { return src.value; }
    //        };
    //
    //        template <ushort_t I>
    //        GT_FUNCTION constexpr int_t sum_dimensions() {
    //            return 0;
    //        }
    //
    //        template <ushort_t I, class T, class... Ts>
    //        GT_FUNCTION constexpr int_t sum_dimensions(T src, Ts... srcs) {
    //            return get_dimension_value_f<I>{}(src) + sum_dimensions<I>(srcs...);
    //        }
    //
    //        template <ushort_t Dim, ushort_t... Is, class... Ts>
    //        GT_FUNCTION constexpr array<int_t, Dim> make_offsets_impl(gt_integer_sequence<ushort_t, Is...>, Ts...
    //        srcs) {
    //            return {sum_dimensions<Is + 1>(srcs...)...};
    //        }
    //
    //        template <ushort_t Dim, class... Ts>
    //        GT_FUNCTION constexpr array<int_t, Dim> make_offsets(Ts... srcs) {
    //            return make_offsets_impl<Dim>(make_gt_integer_sequence<ushort_t, Dim>{}, srcs...);
    //        }
    //    } // namespace _impl

    template <uint_t ID,
        enumtype::intent Intent = enumtype::in,
        typename Extent = extent<0, 0, 0, 0, 0, 0>,
        ushort_t Number = 3>
    struct accessor : accessor_base<make_gt_index_sequence<Number>> {
        using index_t = static_uint<ID>;
        static constexpr enumtype::intent intent = Intent;
        using extent_t = Extent;

        /**inheriting all constructors from accessor_base*/
        using accessor_base<make_gt_index_sequence<Number>>::accessor_base;

        template <uint_t OtherID, typename std::enable_if<ID != OtherID, int>::type = 0>
        GT_FUNCTION constexpr accessor(accessor<OtherID, Intent, Extent, Number> const &src)
            : accessor_base<make_gt_index_sequence<Number>>(src) {}
        //
        //        template <ushort_t I, ushort_t... Is>
        //        GT_FUNCTION constexpr explicit accessor(dimension<I> d, dimension<Is>... ds)
        //            : accessor_base<Number>(make_gt_index_sequence<Number>{}, _impl::make_offsets<Number>(d, ds...)) {
        //            GRIDTOOLS_STATIC_ASSERT((meta::is_set<meta::list<dimension<I>, dimension<Is>...>>::value),
        //                "all dimensions should be of different indicies");
        //        }
    };

    template <uint_t ID, typename Extent = extent<0, 0, 0, 0, 0, 0>, ushort_t Number = 3>
    using in_accessor = accessor<ID, enumtype::in, Extent, Number>;

    template <uint_t ID, typename Extent = extent<0, 0, 0, 0, 0, 0>, ushort_t Number = 3>
    using inout_accessor = accessor<ID, enumtype::inout, Extent, Number>;

} // namespace gridtools
