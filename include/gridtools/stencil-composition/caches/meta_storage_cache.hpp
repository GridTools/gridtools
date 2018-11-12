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

#include "../../common/defs.hpp"
#include "../../common/generic_metafunctions/unzip.hpp"
#include "../../common/gt_assert.hpp"
#include "../../common/host_device.hpp"
#include "../../storage/common/storage_info_interface.hpp" // TODO remove

namespace gridtools {

    template <typename Layout, uint_t... Dims>
    struct meta_storage_cache {

        //        typedef storage_info_interface<0, Layout> meta_storage_t;
        typedef Layout layout_t;
        GRIDTOOLS_STATIC_ASSERT(layout_t::masked_length == sizeof...(Dims),
            GT_INTERNAL_ERROR_MSG("Mismatch in layout length and passed number of dimensions."));

      private:
        template <int LayoutArg>
        struct handle_masked_dims_zero {
            template <typename Dim>
            GT_FUNCTION static constexpr uint_t extend(Dim d) {
                GRIDTOOLS_STATIC_ASSERT(
                    boost::is_integral<Dim>::value, GT_INTERNAL_ERROR_MSG("Dimensions has to be integral type."));
                return error_or_return((d > 0),
                    ((LayoutArg == -1) ? 0 : d),
                    "Tried to instantiate storage info with zero or negative dimensions");
            }
        };

        template <typename>
        struct offset_impl;
        template <int... Is>
        struct offset_impl<layout_map<Is...>> {
            template <size_t... Seq, typename... Ints>
            GT_FUNCTION constexpr uint_t operator()(gt_index_sequence<Seq...>, Ints... idx) const {
                return accumulate(plus_functor(), (idx * stride<Seq>())...);
            }
        };

        template <typename>
        struct padded_total_length_impl;
        template <int... Is>
        struct padded_total_length_impl<layout_map<Is...>> {
            GT_FUNCTION constexpr uint_t operator()() const {
                return accumulate(multiplies(), handle_masked_dims<Is>::extend(Dims)...);
            }
        };

      public:
        GT_FUNCTION
        constexpr meta_storage_cache() {}

        constexpr static uint_t size = accumulate(multiplies(), Dims...); // TODO more

        GT_FUNCTION
        static constexpr uint_t padded_total_length() { return padded_total_length_impl<layout_t>{}(); }

        template <ushort_t Id>
        struct stride_t : std::integral_constant<int_t,
                              get_strides_aux<Layout>::template get_stride<Layout::template at<Id>()>(
                                  handle_masked_dims<Layout::template at<Id>()>::template extend(Dims)...)> {};

        template <ushort_t Id>
        GT_FUNCTION static constexpr int_t stride() {
            return stride_t<Id>::value;
            //            return get_strides_aux<Layout>::template get_stride<Layout::template at<Id>()>(
            //                handle_masked_dims<Layout::template at<Id>()>::extend(Dims)...);
        }

        template <typename... D, typename std::enable_if<is_all_integral<D...>::value, int>::type = 0>
        GT_FUNCTION constexpr int_t index(D... args) const {
            return offset_impl<layout_t>{}(make_gt_index_sequence<sizeof...(Dims)>{}, args...);
        }
    };
} // namespace gridtools
