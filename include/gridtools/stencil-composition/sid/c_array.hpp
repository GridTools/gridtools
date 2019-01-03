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

#include "../../common/integral_constant.hpp"
#include "../../common/tuple.hpp"
#include "../../meta/id.hpp"
#include "../../meta/macros.hpp"
#include "../../meta/push_back.hpp"
#include "../../meta/transform.hpp"
#include "../../meta/type_traits.hpp"

namespace c_array_impl_ {
    template <class T>
    struct get_strides {
        using type = gridtools::tuple<>;
    };

    template <class T>
    struct get_strides<T[]> {
        template <class Stride>
        GT_META_DEFINE_ALIAS(multiply_f,
            gridtools::meta::id,
            (gridtools::integral_constant<ptrdiff_t, std::extent<T>::value * Stride::value>));

        using multiplied_strides_t = GT_META_CALL(
            gridtools::meta::transform, (multiply_f, typename get_strides<T>::type));
        using type = GT_META_CALL(
            gridtools::meta::push_back, (multiplied_strides_t, gridtools::integral_constant<ptrdiff_t, 1>));
    };

    template <class T, size_t N>
    struct get_strides<T[N]> : get_strides<T[]> {};
} // namespace c_array_impl_

template <class T, class Res = gridtools::add_pointer_t<gridtools::remove_all_extents_t<T>>>
constexpr gridtools::enable_if_t<std::is_array<T>::value, Res> sid_get_origin(T &obj) {
    return (Res)obj;
}

template <class T>
constexpr gridtools::enable_if_t<std::is_array<T>::value, typename c_array_impl_::get_strides<T>::type> sid_get_strides(
    T const &) {
    return {};
}
