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
#include "../../common/generic_metafunctions/meta.hpp"
#include "../../common/generic_metafunctions/type_traits.hpp"
#include "../../common/generic_metafunctions/utility.hpp"
#include "../../common/host_device.hpp"

#include "offset.hpp"

namespace gridtools {
    namespace sid {

        /**
         *
         *  PtrOffset is a function of Strides
         *
         *  PtrOffset/PtrOffset arithmetic
         *  Ptr/PtrOffset arithmetic
         *
         *  PtrOffset sid_sum(PtrOffset... offset)
         *
         *  void sid_shift(Ptr& ptr, Strides strides) {
         *  }
         *
         *  auto sid_deref(Ptr ptr, Strides strides) {
         *    return *ptr;
         *  }
         *
         *  void sid_shift(Ptr& ptr, Strides strides, Offsets... offsets) {
         *    ptr += sid_sum((sid_to_ptr_offset<Offsets::index>(strides, offsets)...);
         *  }
         *
         *  auto sid_get_shifted(Ptr ptr, Strides strides, Offsets... offsets) {
         *    return ptr + sid_sum((sid_to_ptr_offset<Offsets::index>(strides, offsets)...);
         *  }
         *
         *  auto sid_deref(Ptr ptr, Strides strides, Offsets... offsets) {
         *    return *(ptr + sid_sum((sid_to_ptr_offset<Offsets::index>(strides, offsets)...));
         *  }
         *
         *
         *   PtrOffset
         *
         */

        struct unused_strides {};

        namespace impl_ {

            struct no_bounds_validator {};

            template <class T>
            struct fake_valid_ptr {
                T m_obj;
                explicit constexpr GT_FUNCTION operator bool() const { return true; }
                constexpr GT_FUNCTION T operator*() const { return m_obj; }
            };

            template <class T, enable_if_t<std::is_lvalue_reference<T>::value, int> = 0>
            constexpr GT_FUNCTION remove_reference_t<T> *make_valid_ptr(T &&src) {
                return &src;
            }

            template <class T, enable_if_t<!std::is_lvalue_reference<T>::value, int> = 0>
            constexpr GT_FUNCTION fake_valid_ptr<T> make_valid_ptr(T &&src) {
                return {src};
            }

            template <class PtrDiff>
            constexpr GT_FUNCTION PtrDiff sid_sum() {
                return {};
            }

            template <class PtrDiff>
            constexpr GT_FUNCTION PtrDiff sid_sum(PtrDiff arg) {
                return arg;
            }

            template <class PtrDiff, class>
            constexpr GT_FUNCTION PtrDiff sid_sum(PtrDiff arg) {
                return arg;
            }

            template <class Sid>
            constexpr GT_FUNCTION no_bounds_validator sid_get_bounds_validator(Sid const &) {
                return {};
            }

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

            // Ptr is a plain pointer => apply shift and do a plain deref
            template <class T, class Strides, class... Offsets>
            constexpr GT_FUNCTION T &sid_deref(T *ptr, Strides const &strides, Offsets... offsets) {
                return *sid_get_shifted(ptr, strides, offsets...);
            }
            template <class T, class Strides, class... Offsets>
            constexpr GT_FUNCTION T *sid_safe_deref(
                T *ptr, Strides const &strides, no_bounds_validator, Offsets... offsets) {
                return sid_get_shifted(ptr, strides, offsets...);
            }

            // if bounds_validator is dummy
            template <class Ptr, class Strides, class... Offsets>
            constexpr GT_FUNCTION auto sid_safe_deref(
                Ptr const &ptr, Strides const &strides, no_bounds_validator, Offsets... offsets)
                GT_AUTO_RETURN(make_valid_ptr(sid_deref(ptr, strides, offsets...)));

            template <class Sid>
            GT_META_DEFINE_ALIAS(ptr_type, meta::id, decltype(sid_get_origin(std::declval<Sid const &>())));

            template <class Sid>
            GT_META_DEFINE_ALIAS(strides_type, meta::id, decltype(sid_get_strides(std::declval<Sid const &>())));

            template <class Sid>
            GT_META_DEFINE_ALIAS(
                bounds_validator_type, meta::id, decltype(sid_get_bounds_validator(std::declval<Sid const &>())));

            template <class... Args>
            GT_META_DEFINE_ALIAS(shift_type, meta::id, decltype(sid_shift(std::declval<Args>()...)));

            template <class... Args>
            GT_META_DEFINE_ALIAS(shifted_type, meta::id, decltype(sid_get_shifted(std::declval<Args>()...)));

            template <class... Args>
            GT_META_DEFINE_ALIAS(deref_type, meta::id, decltype(sid_deref(std::declval<Args>()...)));

            template <class... Args>
            GT_META_DEFINE_ALIAS(safe_deref_type, meta::id, decltype(sid_safe_deref(std::declval<Args>()...)));

            struct offset_model {
                operator int_t() const;
            };
            GRIDTOOLS_STATIC_ASSERT((std::is_convertible<offset_model, int_t>{}), GT_INTERNAL_ERROR);

            template <template <class...> class F, size_t Rank, class... Args>
            GT_META_DEFINE_ALIAS(call_with_rank,
                meta::rename,
                (F, GT_META_CALL(meta::push_front, (GT_META_CALL(meta::repeat_c, (Rank, offset_model)), Args...))));

            template <class Sid,
                size_t DerefRank,
                class PtrType = GT_META_CALL(ptr_type, Sid),
                class StridesType = GT_META_CALL(strides_type, Sid)>
            GT_META_DEFINE_ALIAS(
                reference_type, call_with_rank, (deref_type, DerefRank, PtrType const &, StridesType const &));

            template <class Sid,
                size_t IterRank,
                size_t DerefRank,
                class Ptr = GT_META_CALL(ptr_type, Sid),
                class StridesType = GT_META_CALL(strides_type, Sid),
                class BoundsValidatorType = GT_META_CALL(bounds_validator_type, Sid),
                class ShiftType = GT_META_CALL(call_with_rank, (shift_type, IterRank, Ptr &, StridesType const &)),
                class ShiftedType = GT_META_CALL(
                    call_with_rank, (shifted_type, IterRank, Ptr const &, StridesType const &)),
                class DerefType = GT_META_CALL(reference_type, (Sid, DerefRank)),
                class SafeDerefType = GT_META_CALL(call_with_rank,
                    (safe_deref_type, DerefRank, Ptr const &, StridesType const &, BoundsValidatorType const &))>
            GT_META_DEFINE_ALIAS(is_sid,
                conjunction,
                (std::is_trivially_copyable<Ptr>,
                    std::is_trivially_copyable<StridesType>,
                    std::is_trivially_copyable<BoundsValidatorType>,
                    std::is_void<ShiftType>,
                    std::is_same<ShiftedType, Ptr>,
                    std::is_constructible<bool, SafeDerefType>,
                    std::is_same<decltype(*std::declval<SafeDerefType>()), DerefType>));

            template <class Sid, class Strides = GT_META_CALL(strides_type, Sid)>
            conditional_t<std::is_empty<Strides>::value, Strides, Sid> sid_get_strides_kind(Sid const &);

            template <class Sid, class BoundValidator = GT_META_CALL(bounds_validator_type, Sid)>
            conditional_t<std::is_empty<BoundValidator>::value, BoundValidator, Sid> sid_get_bounds_validator_kind(
                Sid const &);

            template <class Sid>
            GT_META_DEFINE_ALIAS(strides_kind, meta::id, decltype(sid_get_strides_kind(std::declval<Sid const &>())));

            template <class Sid>
            GT_META_DEFINE_ALIAS(
                bounds_validator_kind, meta::id, decltype(sid_get_bounds_validator_kind(std::declval<Sid const &>())));

            template <class Sid>
            constexpr GT_FORCE_INLINE auto get_strides(Sid const &obj) GT_AUTO_RETURN(sid_get_strides(obj));

            template <class T>
            constexpr GT_FORCE_INLINE auto get_bounds_validator(T const &obj)
                GT_AUTO_RETURN(sid_get_bounds_validator(obj));

            template <class Sid>
            constexpr GT_FORCE_INLINE auto get_origin(Sid const &obj) GT_AUTO_RETURN(sid_get_origin(obj));

            template <class Ptr, class Strides, class... Offsets>
            GT_FUNCTION void shift(Ptr &ptr, Strides const &strides, Offsets... offsets) {
                sid_shift(ptr, strides, offsets...);
            }

            template <class Ptr, class Strides, class... Offsets>
            constexpr GT_FUNCTION Ptr get_shifted(Ptr const &ptr, Strides const &strides, Offsets... offsets) {
                return sid_get_shifted(ptr, strides, offsets...);
            }

            template <class Ptr, class Strides, class... Offsets>
            constexpr GT_FUNCTION auto deref(Ptr const &ptr, Strides const &strides, Offsets... offsets)
                GT_AUTO_RETURN(sid_deref(ptr, strides, offsets...));

            template <class Ptr, class Strides, class BoundsChecker, class... Offsets>
            constexpr GT_FUNCTION auto safe_deref(
                Ptr const &ptr, Strides const &strides, BoundsChecker const &bounds_checker, Offsets... offsets)
                GT_AUTO_RETURN(sid_safe_deref(ptr, strides, bounds_checker, offsets...));

            template <class Ref>
            using add_const_to_ref = conditional_t<std::is_reference<Ref>::value,
                add_lvalue_reference_t<add_const_t<remove_reference_t<Ref>>>,
                add_const_t<Ref>>;

            template <class Ptr, class Strides, class... Offsets>
            constexpr GT_FUNCTION auto const_deref(Ptr const &ptr, Strides const &strides, Offsets... offsets)
                -> add_const_to_ref<decltype(sid_deref(ptr, strides, offsets...))> {
                return sid_deref(ptr, strides, offsets...);
            }
        } // namespace impl_

        using impl_::bounds_validator_kind;
        using impl_::strides_kind;

        using impl_::get_bounds_validator;
        using impl_::get_origin;
        using impl_::get_strides;

        using impl_::get_shifted;
        using impl_::shift;

        using impl_::const_deref;
        using impl_::deref;
        using impl_::safe_deref;

        template <class T, size_t IterRank, size_t DerefRank = IterRank, class = void>
        struct is_sid : std::false_type {};

        template <class T, size_t IterRank, size_t DerefRank>
        struct is_sid<T, IterRank, DerefRank, enable_if_t<impl_::is_sid<T, IterRank, DerefRank>::value>>
            : std::true_type {};

        using impl_::bounds_validator_type;
        using impl_::ptr_type;
        using impl_::reference_type;
        using impl_::strides_type;

        template <class Sid, size_t DerefRank, class Ref = reference_type<Sid, DerefRank>>
        GT_META_DEFINE_ALIAS(element_type, meta::id, decay_t<Ref>);
    } // namespace sid

    using sid::is_sid;
} // namespace gridtools