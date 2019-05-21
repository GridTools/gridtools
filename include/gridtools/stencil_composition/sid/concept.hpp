/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <type_traits>

#include "../../common/defs.hpp"
#include "../../common/host_device.hpp"
#include "../../common/hymap.hpp"
#include "../../common/integral_constant.hpp"
#include "../../common/tuple.hpp"
#include "../../common/tuple_util.hpp"
#include "../../meta/defs.hpp"
#include "../../meta/id.hpp"
#include "../../meta/logical.hpp"
#include "../../meta/macros.hpp"
#include "../../meta/push_front.hpp"
#include "../../meta/type_traits.hpp"
#include "simple_ptr_holder.hpp"

/**
 *   Basic API for Stencil Iterable Data (aka SID) concept.
 *
 *   The Concept
 *   ===========
 *
 *   Syntactic part of the concept
 *   -----------------------------
 *
 *   A type `T` models SID concept if it has the following functions defined and available via ADL:
 *     `PtrHolder sid_get_origin(T&);`
 *     `Strides sid_get_strides(T const&);`
 *
 *   The following functions should be declared (definition is not needed) and available via ADL:
 *     `PtrDiff sid_get_ptr_diff(T const&)`
 *     `StridesKind sid_get_strides_kind(T const&);`
 *
 *   The deducible from `T` types `PtrHolder`, `PtrDiff` and `Strides` in their turn should satisfy the constraints:
 *     - `PtrHolder` and `Strides` are trivially copyable
 *     - `PrtHolder` is callable with no args.
 *     - `Ptr` is decayed return type of PtrHolder.
 *     - `PtrDiff` is default constructible
 *     - `Ptr` has `Ptr::operator*() const` which returns non void
 *     - there is `Ptr operator+(Ptr, PtrDiff)` defined
 *     - there is `PtrHolder operator+(PtrHolder, PtrDiff)` defined and `(holder + diff)()` should return the same as
 *       `holder() + diff`
 *     - decayed `Strides` is a hymap
 *
 *   Each type that participate in `Strides` tuple-like (aka `Stride`) should:
 *     - be an integral
 *     or
 *     - be an integral constant (be an instantiation of the `integral_constant` or provide the same functionality)
 *     or
 *     - expressions `ptr += stride * offset` and `ptr_diff += stride * offset` are valid where `ptr`, `ptr_diff` and
 *       `stride` are instances of `Ptr`, `PtrDiff` and `Stride` and `offset` type is integral or instantiation of
 *       integral_constant
 *     or
 *     - the functions `sid_shift(Ptr&, Stride, Offset)` and `sid_shift(PtrDiff&, Stride, Offset)` are defined and
 *       available by ADL;
 *
 *   No constraints on `StridesKind`. It not even has to be complete. (Can be declared but not defined or can be `void`)
 *
 *   Additionally multidimensional C-arrays model `SID` out of the box. For C arrays the outermost dimension goes first.
 *
 *   Semantic part of the concept
 *   ----------------------------
 *   Trivia: pure functional behaviour is expected from provided functions. For example this would be wrong:
 *   ```
 *     int sid_get_stride(my_sid) {
 *       static int count = 0;
 *       return count++;
 *     }
 *   ```
 *
 *   Any SIDs that have the same `StridesKind` would return the equivalent instances from their `sid_get_strides`.
 *   You can think that `sid_get_strides` returns a singleton instance of Strides.
 *
 *   `ptr == ptr + PtrDiff{}`,    for any ptr that is an instance of `Ptr`
 *   `ptr + a + b == ptr + b + a` for any ptr that is an instance of `Ptr` and any a, b of type `PtrDiff`
 *
 *    For concept users: the life time of `Ptr`, `PtrDiff`, `Reference` and `Strides` objects must not exceed
 *    the lifetime of the originated `SID`.
 *    For concept implementors it means that the inferred from the `SID` types can delegate the ownership handling
 *    to `SID`. It is legal for example the `Strides` can be implemented as a reference (or a tuple of references)
 *
 *    TODO(anstaf): formal semantic definition is not complete.
 *
 *    Fallbacks
 *    ---------
 *
 *    `get_strides(Sid)` returns an empty tuple.
 *    `get_ptr_diff(Sid)` returns the same type as `decltype(Ptr{} - Ptr{})`
 *    `get_strides_kind(Sid)` is enabled if `Strides` is empty and returns `Strides`
 *
 *   Compile time API
 *   =================
 *
 *   - is_sid<T> predicate that checks if T models SID syntactically
 *   - sid::ptr_type,
 *     sid::ptr_holder_type,
 *     sid::ptr_diff_type,
 *     sid::strides_type,
 *     sid::strides_kind,
 *     sid::reference_type,
 *     sid::const_reference_type,
 *     sid::element_type - functions in terms of `meta` library. They return various types deducible from the Sid
 *
 *  Run time API
 *  ===========
 *
 *  Wrappers for concept functions:
 *
 *  - Ptr sid::get_origin(Sid&);
 *  - Strides sid::get_strides(Sid const&);
 *  - void sid::shift(T&, Stride, Offset);
 *
 *  Auxiliary functions:
 *
 *  - Stride sid::get_stride<I>(Strides)
 *
 */

namespace gridtools {
    namespace sid {
        namespace concept_impl_ {

            /**
             *   get_static_const_value<T>::value acts as T::value
             *
             *   It is here because in the case where value is a non static member of T, `gcc` generates an error
             *   for `T::value` even in the substitution context. I.e `gcc` fails to follow SFINAE principle.
             *   [Checked for gcc 8.2]
             *
             */
            template <class T, class = void>
            struct get_static_const_value;
            template <class T>
            struct get_static_const_value<T,
                enable_if_t<std::is_integral<decltype(T::value)>::value && std::is_const<decltype(T::value)>::value &&
                            !std::is_member_pointer<decltype(&T::value)>::value>>
                : std::integral_constant<decltype(T::value), T::value> {};

            /**
             *  generic trait for integral constant
             *
             *  TODO(anstaf) : consider moving it to `meta` or to `common`
             */
            template <class T, class = void>
            struct is_integral_constant : std::false_type {};
            template <class T>
            struct is_integral_constant<T,
                enable_if_t<std::is_empty<T>::value && std::is_integral<decltype(T::value)>::value &&
                            std::is_convertible<T, typename get_static_const_value<T>::value_type>::value &&
                            T() == get_static_const_value<T>::value>> : std::true_type {};

            /**
             *  generic trait for integral constant of Val
             *
             *  TODO(anstaf) : consider moving it to `meta` or to `common`
             */
            template <class T, int Val, class = void>
            struct is_integral_constant_of : std::false_type {};

            template <class T, int Val>
            struct is_integral_constant_of<T, Val, enable_if_t<is_integral_constant<T>::value && T() == Val>>
                : std::true_type {};

            /////// BEGIN defaults PART /////
            template <class...>
            struct default_strides_templ;
            template <>
            struct default_strides_templ<> {};
            using default_strides = default_strides_templ<>;

            template <class Ptr>
            auto sid_get_default_ptr_diff(Ptr const &ptr) -> decltype(ptr - ptr);

            template <class Ptr>
            using default_ptr_diff = decltype(::gridtools::sid::concept_impl_::sid_get_default_ptr_diff(
                std::declval<add_lvalue_reference_t<add_const_t<Ptr>>>()));

            template <class T, class = void>
            struct is_empty_or_tuple_of_empties : std::is_empty<T> {};

            template <class Tup, class Types = tuple_util::traits::to_types<Tup>>
            using is_tuple_of_empties = meta::all_of<is_empty_or_tuple_of_empties, Types>;

            template <class Tup>
            struct is_empty_or_tuple_of_empties<Tup, enable_if_t<is_tuple_of_empties<Tup>::value>> : std::true_type {};

            namespace lazy {
                template <class, class = void>
                struct default_kind;
                template <class T>
                struct default_kind<T, enable_if_t<is_empty_or_tuple_of_empties<decay_t<T>>::value>> : std::decay<T> {};
            } // namespace lazy
            GT_META_DELEGATE_TO_LAZY(default_kind, class T, T);
            /////// END defaults PART ///////

            /////// Fallbacks

            struct not_provided;

            not_provided sid_get_strides(...);
            not_provided sid_get_ptr_diff(...);
            not_provided sid_get_strides_kind(...);

            // BEGIN `get_origin` PART

            /**
             *  `get_origin` delegates to `sid_get_origin`
             */
            template <class Sid>
            GT_CONSTEXPR auto get_origin(Sid &obj) -> decltype(sid_get_origin(obj)) {
                return sid_get_origin(obj);
            }

            /**
             *  C-array specialization
             */
            template <class T, class Res = gridtools::add_pointer_t<gridtools::remove_all_extents_t<T>>>
            GT_CONSTEXPR gridtools::enable_if_t<std::is_array<T>::value, host_device::simple_ptr_holder<Res>>
            get_origin(T &obj) {
                return {(Res)obj};
            }

            /**
             *  `Ptr Holder` type is deduced from `get_origin`
             */
            template <class Sid>
            using ptr_holder_type = decltype(::gridtools::sid::concept_impl_::get_origin(std::declval<Sid &>()));

            /**
             *  `Ptr` type is deduced from `get_origin`
             */
            template <class Sid, class PtrHolder = ptr_holder_type<Sid>>
            using ptr_type = decay_t<decltype(std::declval<PtrHolder>()())>;

            /**
             *  `Reference` type is deduced from `Ptr` type
             */
            template <class Sid, class Ptr = ptr_type<Sid>>
            using reference_type = decltype(*std::declval<Ptr const &>());

            // END `get_origin` PART

            // BEGIN `ptr_diff_type`

            /**
             *  a proxy for sid_get_ptr_diff ADL resolution
             */
            template <class Sid, class Res = decltype(sid_get_ptr_diff(std::declval<Sid const &>()))>
            enable_if_t<!std::is_same<Res, not_provided>::value, Res> get_ptr_diff(Sid const &);

            template <class Sid, class Res = decltype(sid_get_ptr_diff(std::declval<Sid const &>()))>
            enable_if_t<std::is_same<Res, not_provided>::value, default_ptr_diff<ptr_type<Sid>>> get_ptr_diff(
                Sid const &);

            /**
             *  `PtrDiff` type is deduced from `get_ptr_diff`
             */
            template <class Sid>
            using ptr_diff_type = decltype(::gridtools::sid::concept_impl_::get_ptr_diff(std::declval<Sid const &>()));

            // END `ptr_diff_type`

            // BEGIN `get_strides` PART

            /**
             *  `get_strides` delegates to `sid_get_strides`
             */
            template <class Sid, class Res = decltype(sid_get_strides(std::declval<Sid const &>()))>
            GT_CONSTEXPR enable_if_t<!std::is_same<Res, not_provided>::value && !std::is_array<Sid>::value, Res>
            get_strides(Sid const &obj) {
                return sid_get_strides(obj);
            }

            template <class Sid, class Res = decltype(sid_get_strides(std::declval<Sid const &>()))>
            GT_CONSTEXPR
                enable_if_t<std::is_same<Res, not_provided>::value && !std::is_array<Sid>::value, default_strides>
                get_strides(Sid const &) {
                return {};
            }

            template <class T, size_t ElemSize = sizeof(remove_all_extents_t<T>)>
            struct get_array_strides {
                using type = tuple<>;
            };

            template <class Inner, size_t ElemSize>
            struct get_array_strides<Inner[], ElemSize> {
                using type = meta::push_front<typename get_array_strides<Inner, ElemSize>::type,
                    integral_constant<ptrdiff_t, sizeof(Inner) / ElemSize>>;
            };

            template <class Inner, size_t N, size_t ElemSize>
            struct get_array_strides<Inner[N], ElemSize> : get_array_strides<Inner[], ElemSize> {};

            template <class T>
            GT_CONSTEXPR enable_if_t<std::is_array<T>::value, typename get_array_strides<T>::type> get_strides(
                T const &) {
                return {};
            }

            /**
             *  `Strides` type is deduced from `get_strides`
             */
            template <class Sid>
            using strides_type = decltype(::gridtools::sid::concept_impl_::get_strides(std::declval<Sid const &>()));

            // END `get_strides` PART

            // BEGIN `strides_kind` PART

            template <class Sid, class Res = decltype(sid_get_strides_kind(std::declval<Sid const &>()))>
            enable_if_t<!std::is_same<Res, not_provided>::value, Res> get_strides_kind(Sid const &);

            template <class Sid, class Res = decltype(sid_get_strides_kind(std::declval<Sid const &>()))>
            enable_if_t<std::is_same<Res, not_provided>::value, default_kind<strides_type<Sid>>> get_strides_kind(
                Sid const &);

            /**
             *  `strides_kind` is deduced from `get_strides_kind`
             */
            template <class Sid>
            using strides_kind =
                decltype(::gridtools::sid::concept_impl_::get_strides_kind(std::declval<Sid const &>()));

            // END `strides_kind` PART

            // BEGIN `shift` PART

            // no fallback for `sid_shift`

            /**
             *  Predicate that determines if `shift` needs to be apply
             *
             *  If stride of offset are zero or the target has no state, we don't need to shift
             */
            template <class T, class Stride, class Offset>
            using need_shift = bool_constant<!(std::is_empty<T>::value || is_integral_constant_of<Stride, 0>::value ||
                                               is_integral_constant_of<Offset, 0>::value)>;

            /**
             *  additional proxy is used here to ensure that evaluation context of `obj += stride * offset`
             *  is always the same.
             */
            template <class T, class Strides>
            auto default_shift(T &obj, Strides const &stride, int offset = 0) -> decltype(obj += stride * offset);

            /**
             *  true if we can do implement shift as `obj += stride * offset`
             */
            template <class T, class Strides, class = void>
            struct is_default_shiftable : std::false_type {};
            template <class T, class Stride>
            struct is_default_shiftable<T,
                Stride,
                void_t<decltype(::gridtools::sid::concept_impl_::default_shift(
                    std::declval<T &>(), std::declval<Stride const &>()))>> : std::true_type {};

            template <class T>
            auto inc_operator(T &obj) -> decltype(++obj);

            /**
             *  True if T has operator++
             */
            template <class T, class = void>
            struct has_inc : std::false_type {};
            template <class T>
            struct has_inc<T, void_t<decltype(::gridtools::sid::concept_impl_::inc_operator(std::declval<T &>()))>>
                : std::true_type {};

            template <class T>
            auto dec_operator(T &obj) -> decltype(--obj);

            /**
             *  True if T has operator--
             */
            template <class T, class = void>
            struct has_dec : std::false_type {};
            template <class T>
            struct has_dec<T, void_t<decltype(::gridtools::sid::concept_impl_::dec_operator(std::declval<T &>()))>>
                : std::true_type {};

            template <class T, class Arg>
            auto dec_assignment_operator(T &obj, Arg const &arg) -> decltype(obj = -arg);

            /**
             *  True if T has operator-=
             */
            template <class T, class Arg, class = void>
            struct has_dec_assignment : std::false_type {};
            template <class T, class Arg>
            struct has_dec_assignment<T,
                Arg,
                void_t<decltype(::gridtools::sid::concept_impl_::dec_assignment_operator(
                    std::declval<T &>(), std::declval<Arg const &>()))>> : std::true_type {};

            /**
             *  noop `shift` overload
             */
            template <class T, class Stride, class Offset>
            GT_FUNCTION enable_if_t<!need_shift<T, Stride, Offset>::value> shift(T &, Stride const &, Offset const &) {}

            /**
             * `shift` overload that delegates to `sid_shift`
             *
             *  Enabled only if shift can not be implemented as `obj += stride * offset`
             */
            template <class T,
                class Stride,
                class Offset,
                enable_if_t<need_shift<T, Stride, Offset>::value && !is_default_shiftable<T, Stride>::value, int> = 0>
            GT_FUNCTION decltype(auto) shift(
                T &obj, Stride const &GT_RESTRICT stride, Offset const &GT_RESTRICT offset) {
                return sid_shift(obj, stride, offset);
            }

            /**
             *  `shift` overload where both `stride` and `offset` are known in compile time
             */
            template <class T,
                class Stride,
                class Offset,
                int_t PtrOffset = get_static_const_value<Stride>::value *Offset::value>
            GT_FUNCTION enable_if_t<need_shift<T, Stride, Offset>::value && is_default_shiftable<T, Stride>::value &&
                                    !(has_inc<T>::value && PtrOffset == 1) && !(has_dec<T>::value && PtrOffset == -1)>
            shift(T &obj, Stride const &, Offset const &) {
                obj += integral_constant<int_t, PtrOffset>{};
            }

            /**
             *  `shift` overload where the stride and offset are both 1 (or both -1)
             */
            template <class T,
                class Stride,
                class Offset,
                int_t PtrOffset = get_static_const_value<Stride>::value *Offset::value>
            GT_FUNCTION enable_if_t<need_shift<T, Stride, Offset>::value && is_default_shiftable<T, Stride>::value &&
                                    has_inc<T>::value && PtrOffset == 1>
            shift(T &obj, Stride const &, Offset const &) {
                ++obj;
            }

            /**
             *  `shift` overload where the stride are offset are both 1, -1 (or -1, 1)
             */
            template <class T,
                class Stride,
                class Offset,
                int_t PtrOffset = get_static_const_value<Stride>::value *Offset::value>
            GT_FUNCTION enable_if_t<need_shift<T, Stride, Offset>::value && is_default_shiftable<T, Stride>::value &&
                                    has_dec<T>::value && PtrOffset == -1>
            shift(T &obj, Stride const &, Offset const &) {
                --obj;
            }

            /**
             *  `shift` overload where the offset is 1
             */
            template <class T, class Stride, class Offset>
            GT_FUNCTION enable_if_t<need_shift<T, Stride, Offset>::value && is_default_shiftable<T, Stride>::value &&
                                    !is_integral_constant<Stride>::value && is_integral_constant_of<Offset, 1>::value>
            shift(T &obj, Stride const &GT_RESTRICT stride, Offset const &) {
                obj += stride;
            }

            /**
             *  `shift` overload where the offset is -1
             */
            template <class T, class Stride, class Offset>
            GT_FUNCTION enable_if_t<need_shift<T, Stride, Offset>::value && is_default_shiftable<T, Stride>::value &&
                                    !is_integral_constant<Stride>::value &&
                                    is_integral_constant_of<Offset, -1>::value && has_dec_assignment<T, Stride>::value>
            shift(T &obj, Stride const &GT_RESTRICT stride, Offset const &) {
                obj -= stride;
            }

            /**
             *  `shift` overload where the stride is 1
             */
            template <class T, class Stride, class Offset>
            GT_FUNCTION enable_if_t<need_shift<T, Stride, Offset>::value && is_default_shiftable<T, Stride>::value &&
                                    is_integral_constant_of<Stride, 1>::value && !is_integral_constant<Offset>::value>
            shift(T &obj, Stride const &, Offset const &GT_RESTRICT offset) {
                obj += offset;
            }

            /**
             *  `shift` overload where the stride is -1
             */
            template <class T, class Stride, class Offset>
            GT_FUNCTION enable_if_t<need_shift<T, Stride, Offset>::value && is_default_shiftable<T, Stride>::value &&
                                    is_integral_constant_of<Stride, -1>::value &&
                                    !is_integral_constant<Offset>::value && has_dec_assignment<T, Stride>::value>
            shift(T &obj, Stride const &, Offset const &GT_RESTRICT offset) {
                obj -= offset;
            }

            /**
             *  `shift` overload, default version
             */
            template <class T, class Stride, class Offset>
            GT_FUNCTION
                enable_if_t<need_shift<T, Stride, Offset>::value && is_default_shiftable<T, Stride>::value &&
                            !(is_integral_constant<Stride>::value && is_integral_constant<Offset>::value) &&
                            !(is_integral_constant_of<Stride, 1>::value || is_integral_constant_of<Offset, 1>::value) &&
                            !(has_dec_assignment<T, Stride>::value && (is_integral_constant_of<Stride, -1>::value ||
                                                                          is_integral_constant_of<Offset, -1>::value))>
                shift(T &obj, Stride const &GT_RESTRICT stride, Offset const &GT_RESTRICT offset) {
                obj += stride * offset;
            }

            // END `shift` PART

            /**
             *  Meta predicate that validates a single stride type against `shift` function
             */
            template <class T>
            struct is_valid_stride {
                template <class Stride, class = void>
                struct apply : std::false_type {};
                template <class Stride>
                struct apply<Stride,
                    void_t<decltype(::gridtools::sid::concept_impl_::shift(
                        std::declval<T &>(), std::declval<Stride &>(), int_t{}))>> : std::true_type {};
            };

            /**
             *  Meta predicate that validates a list of stride type against `shift` function
             */
            template <class StrideTypes, class T>
            using are_valid_strides = meta::all_of<is_valid_stride<T>::template apply, StrideTypes>;

            /**
             *  Sfinae unsafe version of `is_sid` predicate
             */
            template <class Sid,
                // Extracting all the derived types from Sid
                class PtrHolder = ptr_holder_type<Sid>,
                class Ptr = ptr_type<Sid>,
                class ReferenceType = reference_type<Sid>,
                class PtrDiff = ptr_diff_type<Sid>,
                class StridesType = strides_type<Sid>,
                class StrideTypeList = tuple_util::traits::to_types<decay_t<StridesType>>,
                class StridesKind = strides_kind<Sid>>
            using is_sid = conjunction<std::is_trivially_copyable<PtrHolder>,
                std::is_trivially_copyable<StridesType>,

                // verify that `PtrDiff` is sane
                std::is_default_constructible<PtrDiff>,
                std::is_convertible<decltype(std::declval<Ptr const &>() + std::declval<PtrDiff const &>()), Ptr>,

                // `PtrHolder` supports `+` as well
                // TODO(anstaf): we can do better here: verify that if we transform PtrHolder that way the result
                //               thing also models SID
                std::is_same<decay_t<decltype(std::declval<PtrHolder const &>() + std::declval<PtrDiff const &>())>,
                    PtrHolder>,

                // verify that `Reference` is sane
                negation<std::is_void<ReferenceType>>,

                // all strides must be applied via `shift` with both `Ptr` and `PtrDiff`
                are_valid_strides<StrideTypeList, Ptr>,
                are_valid_strides<StrideTypeList, PtrDiff>>;

        } // namespace concept_impl_

        // Meta functions

#if GT_BROKEN_TEMPLATE_ALIASES
#define GT_SID_DELEGATE_FROM_IMPL(name) \
    template <class Sid>                \
    struct name : meta::id<concept_impl_::name<Sid>> {}
#else
#define GT_SID_DELEGATE_FROM_IMPL(name) using concept_impl_::name
#endif

        GT_SID_DELEGATE_FROM_IMPL(ptr_holder_type);
        GT_SID_DELEGATE_FROM_IMPL(ptr_type);
        GT_SID_DELEGATE_FROM_IMPL(ptr_diff_type);
        GT_SID_DELEGATE_FROM_IMPL(reference_type);
        GT_SID_DELEGATE_FROM_IMPL(strides_type);
        GT_SID_DELEGATE_FROM_IMPL(strides_kind);

        GT_SID_DELEGATE_FROM_IMPL(default_ptr_diff);

#undef GT_SID_DELEGATE_FROM_IMPL

        // Runtime functions
        using concept_impl_::get_origin;
        using concept_impl_::get_strides;
        using concept_impl_::shift;

        // Default behaviour
        using concept_impl_::default_kind;
        using concept_impl_::default_strides;
        using default_stride = integral_constant<int_t, 0>;

        /**
         *  Does a type models the SID concept
         */
        template <class T, class = void>
        struct is_sid : std::false_type {};
        template <class T>
        struct is_sid<T, enable_if_t<concept_impl_::is_sid<T>::value>> : std::true_type {};

        // Auxiliary API

        /**
         *  The type of the element of the SID
         */
        template <class Sid, class Ref = reference_type<Sid>>
        using element_type = meta::id<decay_t<Ref>>;

        /**
         *  The const variation of the reference type
         */
        template <class Sid, class Ref = reference_type<Sid>>
        using const_reference_type = meta::id<conditional_t<std::is_reference<Ref>::value,
            add_lvalue_reference_t<add_const_t<remove_reference_t<Ref>>>,
            add_const_t<Ref>>>;

        /**
         *  A getter from Strides to the given stride.
         *
         *  If `Stride` doesn't have `Key`, integral_constant<int_t, 0> is returned.
         *  Which allows to silently ignore the offsets in non existing dimensions.
         */
        template <class Key, class Strides>
        GT_CONSTEXPR GT_FUNCTION decltype(auto) get_stride(Strides &&strides) {
            return gridtools::host_device::at_key_with_default<Key, default_stride>(strides);
        }

        struct get_origin_f {
            template <class T>
            GT_CONSTEXPR auto operator()(T &obj) const {
                return get_origin(obj);
            }
        };
    } // namespace sid

    /*
     *  Promote `is_sid` one level up.
     *
     *  Just because `sid::is_sid` looks a bit redundant
     */
    using sid::is_sid;
} // namespace gridtools
