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

#include "../../common/generic_metafunctions/meta.hpp"
#include "../../common/generic_metafunctions/type_traits.hpp"
#include "../../common/generic_metafunctions/utility.hpp"
#include "../../common/host_device.hpp"

namespace gridtools {
    namespace sid {
        struct empty_strides {};

        struct empty_bounds_validator {
            template <class Ptr>
            constexpr GT_FORCE_INLINE bool operator()(Ptr &&) const {
                return true;
            }
        };

        namespace impl_ {
            template <class Sid>
            constexpr GT_FORCE_INLINE empty_strides sid_get_strides(Sid const &) {
                return {};
            }

            template <class Sid>
            constexpr GT_FORCE_INLINE empty_bounds_validator sid_get_bounds_validator(Sid const &) {
                return {};
            }

            template <class Sid>
            meta::lazy::id<Sid> sid_get_kind(Sid const &);

            template <class Ptr, class Strides, class... Offsets>
            GT_FUNCTION Ptr sid_get_shifted(Ptr ptr, Strides const &strides, Offsets... offsets) {
                sid_shift(ptr, strides, offsets...);
                return ptr;
            }

            template <class Ptr,
                class Strides,
                class... Offsets,
                enable_if_t<std::is_move_assignable<Ptr>::value, int> = 0>
            GT_FUNCTION void sid_shift(Ptr &ptr, Strides const &strides, Offsets... offsets) {
                ptr = sid_get_shifted(ptr, strides, offsets...);
            }

            template <class T, class Strides>
            constexpr GT_FUNCTION T &sid_deref(T *ptr, Strides const &) {
                return *ptr;
            }

            template <class Ptr, class Strides, class... Offsets>
            constexpr GT_FUNCTION auto sid_deref(Ptr const &ptr, Strides const &strides, Offsets... offsets)
                GT_AUTO_RETURN(sid_deref(sid_get_shifted(ptr, strides, offsets...), strides));

            template <class T>
            constexpr GT_FORCE_INLINE auto get_strides(T const &obj) GT_AUTO_RETURN(sid_get_strides(obj));

            template <class T>
            constexpr GT_FORCE_INLINE auto get_zero_ptr(T const &obj) GT_AUTO_RETURN(sid_get_zero_ptr(obj));

            template <class T>
            constexpr GT_FORCE_INLINE auto get_bounds_validator(T const &obj)
                GT_AUTO_RETURN(sid_get_bounds_validator(obj));

            template <class Ptr, class Strides, class... Offsets>
            constexpr GT_FUNCTION Ptr get_shifted(Ptr const &ptr, Strides const &strides, Offsets... offsets) {
                return sid_get_shifted(ptr, strides, offsets...);
            }

            template <class Ptr, class Strides, class... Offsets>
            GT_FUNCTION void shift(Ptr &ptr, Strides const &strides, Offsets... offsets) {
                sid_shift(ptr, strides, offsets...);
            }

            template <class Ptr, class Strides, class... Offsets>
            constexpr GT_FUNCTION auto deref(Ptr const &ptr, Strides const &strides, Offsets... offsets)
                GT_AUTO_RETURN(sid_deref(ptr, strides, offsets...));

            template <class Ref>
            using add_const_to_ref = conditional_t<std::is_reference<Ref>::value,
                add_lvalue_reference_t<add_const_t<remove_reference_t<Ref>>>,
                add_const_t<Ref>>;

            template <class Ptr, class Strides, class... Offsets>
            constexpr GT_FUNCTION auto const_deref(Ptr const &ptr, Strides const &strides, Offsets... offsets)
                -> add_const_to_ref<decltype(sid_deref(ptr, strides, offsets...))> {
                return sid_deref(ptr, strides, offsets...);
            }

            template <class Sid>
            GT_META_DEFINE_ALIAS(kind, meta::id, decltype(sid_get_kind(std::declval<Sid const &>())));

            template <class T>
            using ptr_type = decltype(sid_get_zero_ptr(std::declval<T const &>()));

            template <class T>
            using strides_type = decltype(sid_get_strides(std::declval<T const &>()));

        } // namespace impl_

        using impl_::kind;

        using impl_::const_deref;
        using impl_::deref;
        using impl_::get_bounds_validator;
        using impl_::get_shifted;
        using impl_::get_strides;
        using impl_::get_zero_ptr;
        using impl_::shift;

        template <class Sid>
        GT_META_DEFINE_ALIAS(ptr_type, meta::id, decltype(get_zero_ptr(std::declval<Sid const &>())));

        template <class Sid>
        GT_META_DEFINE_ALIAS(strides_type, meta::id, decltype(get_strides(std::declval<Sid const &>())));

        template <class Sid, class... ExtraArgs>
        GT_META_DEFINE_ALIAS(reference_type,
            meta::id,
            decltype(deref(get_zero_ptr(std::declval<Sid const &>()), std::declval<ExtraArgs>()...)));

        template <class Sid, class... ExtraArgs>
        GT_META_DEFINE_ALIAS(element_type,
            meta::id,
            decay_t<decltype(deref(get_zero_ptr(std::declval<Sid const &>()), std::declval<ExtraArgs>()...))>);

        template <class T, size_t IterationRank, size_t DerefRank = IterationRank>
        using is_sid = std::false_type;
    } // namespace sid

    using sid::is_sid;
} // namespace gridtools