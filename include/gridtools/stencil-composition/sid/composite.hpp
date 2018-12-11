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

#include "../../common/binops.hpp"
#include "../../common/defs.hpp"
#include "../../common/functional.hpp"
#include "../../common/generic_metafunctions/for_each.hpp"
#include "../../common/generic_metafunctions/utility.hpp"
#include "../../common/host_device.hpp"
#include "../../common/tuple.hpp"
#include "../../common/tuple_util.hpp"
#include "../../meta.hpp"
#include "concept.hpp"

namespace gridtools {
    namespace sid {
        namespace impl_ {
            struct deref_f {
                template <class T>
                constexpr GT_FUNCTION auto operator()(T const &obj) const GT_AUTO_RETURN(*obj);
            };

            struct call_with_f {
                template <class Fun, class... Args>
                constexpr GT_FUNCTION auto operator()(Fun &&fun, Args &&... args) const
                    GT_AUTO_RETURN(const_expr::forward<Fun>(fun)(const_expr::forward<Args>(args)...));
            };

            template <class... Ptrs>
            struct composite_ptr {
                tuple<Ptrs...> m_vals;
                GT_TUPLE_UTIL_FORWARD_GETTER_TO_MEMBER(composite_ptr, m_vals);
                GT_TULPE_UTIL_FORWARD_CTORS_TO_MEMBER(composite_ptr, m_vals);
                constexpr auto operator*() const GT_AUTO_RETURN(tuple_util::host_device::transform(deref_f{}, m_vals));
            };

            template <class State,
                class Kind,
                class Map = GT_META_CALL(meta::first, State),
                class Kinds = GT_META_CALL(meta::second, State),
                class Pos = typename meta::length<Map>::type,
                class KindsPos = typename meta::find<Kinds, Kind>::type,
                class NewMap = GT_META_CALL(meta::push_back, (Map, meta::list<Pos, KindsPos>)),
                class NewKinds = GT_META_CALL(meta::if_c,
                    (KindsPos::value == meta::length<Kinds>::value,
                        GT_META_CALL(meta::push_back, (Kinds, Kind)),
                        Kinds))>
            GT_META_DEFINE_ALIAS(make_map_helper, meta::list, (NewMap, NewKinds));

            template <class Kinds>
            GT_META_DEFINE_ALIAS(make_index_map,
                meta::first,
                (GT_META_CALL(meta::lfold, (make_map_helper, meta::list<meta::list<>, meta::list<>>, Kinds))));

            template <class T>
            constexpr GT_FUNCTION auto maybe_equal(T const &lhs, T const &rhs) GT_AUTO_RETURN(lhs == rhs);

            constexpr GT_FUNCTION bool maybe_equal(...) { return true; }

            template <class>
            struct item_generator;

            template <template <class...> class L, class Key, class PrimaryIndex, class... SecondaryIndices>
            struct item_generator<L<Key, PrimaryIndex, SecondaryIndices...>> {
                using type = item_generator;

                template <class Args, class Res = GT_META_CALL(tuple_util::element, (PrimaryIndex::value, Args))>
                constexpr GT_FUNCTION Res const &operator()(Args const &args) const noexcept {
                    GRIDTOOLS_STATIC_ASSERT(
                        (conjunction<std::is_same<GT_META_CALL(tuple_util::element, (SecondaryIndices::value, Args)),
                                Res>...>::value),
                        GT_INTERNAL_ERROR);
                    return (void)(int[]){
                               (assert(maybe_equal(tuple_util::host_device::get<SecondaryIndices::value>(args),
                                    tuple_util::host_device::get<PrimaryIndex::value>(args))),
                                   0)...},
                           tuple_util::host_device::get<PrimaryIndex::value>(args);
                }
            };

            template <class ObjTup, class StrideTup, class Offset>
            struct shift_t {
                ObjTup &RESTRICT m_obj_tup;
                StrideTup const &RESTRICT m_stride_tup;
                Offset const &m_offset;

                template <class I>
                GT_FUNCTION void operator()() const {
                    shift(tuple_util::get<I::value>(m_obj_tup), tuple_util::get<I::value>(m_stride_tup), m_offset);
                }
            };

            template <class ObjTup, class StrideTup, class Offset>
            void composite_shift_impl(
                ObjTup &RESTRICT obj_tup, StrideTup const &RESTRICT stride_tup, Offset const &offset) {
                static constexpr size_t size = tuple_util::size<ObjTup>::value;
                GRIDTOOLS_STATIC_ASSERT(tuple_util::size<StrideTup>::value == size, GT_INTERNAL_ERROR);
                host_device::for_each_type<GT_META_CALL(meta::make_indices_c, size)>(
                    shift_t<ObjTup, StrideTup, Offset>{obj_tup, stride_tup, offset});
            }

            template <class Map>
            struct compressed {
                using inversed_map_t = GT_META_CALL(meta::mp_inverse, Map);
                using primary_indices_t = GT_META_CALL(meta::transform, (meta::second, inversed_map_t));

                using generators_t = GT_META_CALL(meta::transform, (item_generator, inversed_map_t));

                template <size_t I,
                    class T,
                    class Item = GT_META_CALL(meta::mp_find, (Map, std::integral_constant<size_t, I>)),
                    class Pos = GT_META_CALL(meta::second, Item)>
                static constexpr GT_FUNCTION auto get(T &&obj)
                    GT_AUTO_RETURN(tuple_util::host_device::get<Pos::value>(const_expr::forward<T>(obj).m_vals));

                template <class... Ts>
                struct composite {
                    template <class I>
                    GT_META_DEFINE_ALIAS(get_compressed_type, meta::at, (meta::list<Ts...>, I));

                    using vals_t = GT_META_CALL(meta::rename,
                        (meta::ctor<tuple<>>::apply,
                            GT_META_CALL(meta::transform, (get_compressed_type, primary_indices_t))));

                    vals_t m_vals;

                    template <class... Args, enable_if_t<sizeof...(Args) == sizeof...(Ts), int> = 0>
                    constexpr GT_FUNCTION composite(Args &&... args) noexcept
                        : composite(tuple<Args &&...>{const_expr::forward<Args &&>(args)...}) {}

                    template <class... Args, enable_if_t<sizeof...(Args) == sizeof...(Ts), int> = 0>
                    constexpr GT_FUNCTION composite(tuple<Args...> &&tup) noexcept
                        : m_vals{tuple_util::host_device::generate<generators_t, vals_t>(const_expr::move(tup))} {}

                    composite() = default;
                    composite(composite const &) = default;
                    composite(composite &&) noexcept = default;
                    composite &operator=(composite const &) = default;
                    composite &operator=(composite &&) = default;

                    template <class Tup>
                    GT_FUNCTION bool operator()(Tup const &tup) const {
                        return tuple_util::host_device::all_of(call_with_f{}, *this, tup);
                    }

                    friend compressed tuple_getter(composite) {}

                    template <class... Ptrs>
                    friend constexpr GT_FUNCTION composite_ptr<Ptrs...> operator+(
                        composite_ptr<Ptrs...> const &lhs, composite const &rhs) {
                        return tuple_util::host_device::transform(binop::sum{}, lhs, rhs);
                    }

                    template <class... Ptrs, class Offset>
                    friend GT_FUNCTION void sid_shift(
                        composite_ptr<Ptrs...> &ptr, composite const &stride, Offset const &offset) {
                        composite_shift_impl(ptr.m_vals, stride, offset);
                    }
                };

                template <class... PtrDiffs, class... Strides, class Offset>
                friend GT_FUNCTION void sid_shift(composite<PtrDiffs...> &RESTRICT ptr_diff,
                    composite<Strides...> const &RESTRICT stride,
                    Offset const &offset) {
                    composite_shift_impl(ptr_diff.m_vals, stride.m_vals, offset);
                }
            };

            constexpr size_t max_impl(size_t acc, size_t const *cur, size_t const *last) {
                return cur == last ? acc : max_impl(*cur > acc ? *cur : acc, cur + 1, last);
            }

            template <size_t N>
            constexpr size_t max(size_t const (&vals)[N]) {
                return max_impl(0, vals, vals + N);
            }

            template <class I, class Strides>
#if GT_BROKEN_TEMPLATE_ALIASES
            struct normalized_stride_type : std::conditional<(I::value < tuple_util::size<Strides>::value),
                                                tuple_util::lazy::element<I::value, Strides>,
                                                meta::lazy::id<default_stride>>::type {
            };
#else
            using normalized_stride_type = typename std::conditional<(I::value < tuple_util::size<Strides>::value),
                tuple_util::lazy::element<I::value, Strides>,
                meta::lazy::id<default_stride>>::type::type;
#endif

            template <class Is>
            struct normalize_strides_f;

            template <template <class...> class L, class... Is>
            struct normalize_strides_f<L<Is...>> {
                template <class Strides>
                constexpr GT_FUNCTION auto operator()(Strides &&strides) const
                    GT_AUTO_RETURN(tuple_util::host_device::make<tuple>(
                        get_stride<Is::value>(const_expr::forward<Strides>(strides))...));
            };

            template <class StrideIndices>
            struct get_normalized_strides_f {
                template <class Sid>
                constexpr GT_FUNCTION auto operator()(Sid const &sid) const
                    GT_AUTO_RETURN(normalize_strides_f<StrideIndices>{}(get_strides(sid)));
            };

            struct converter_f {
                template <class Src, class DstId, class Res = typename DstId::type>
                constexpr GT_FUNCTION Res operator()(Src &&src, DstId &&) const {
                    return Res{const_expr::forward<Src>(src)};
                }
            };
        } // namespace impl_

        /**
         *  This class models both `SID` and `tuple_like` concepts at the same time
         *
         *  All derived types of the `composite` are `tuple_like`s of the correspondent types of the original ones.
         *  Example:
         *    say you have two sids: `s1` and `s2` of types `S1` and `S2`
         *    you can compose them: `comoposite<S1, S2> c = {s1, s2};`
         *    now `c` is a `SID` as well, you can call `get_origin(c)`, `get_strides(c)` etc.
         *    also yo have an access to `s1` and `s2` via `tuple_util::get`: `tuple_util::get<0>(c)` is the same as `s1`
         *
         *    When composing strides together the maximum strides size is calculated and for the original strides are
         *    expanded to that maximum with `integral_constant<int_t, 0>`.
         *    Internaly `composite` utilizes the fact that the strides of the same kind are always the same. This  means
         *    that the composite stride holds doesn't hold duplicate strides.
         *
         *  @tparam Sids - all of them should model `SID` concept
         */
        template <class... Sids>
        class composite {
            GRIDTOOLS_STATIC_ASSERT(conjunction<is_sid<Sids>...>::value, GT_INTERNAL_ERROR);

            tuple<Sids...> m_sids;

            using strides_kinds_t = meta::list<GT_META_CALL(strides_kind, Sids)...>;
            using bounds_validator_kinds_t = meta::list<GT_META_CALL(bounds_validator_kind, Sids)...>;

            using strides_map_t = GT_META_CALL(impl_::make_index_map, strides_kinds_t);
            using bounds_validator_map_t = GT_META_CALL(impl_::make_index_map, bounds_validator_kinds_t);

            using stride_indices_t = GT_META_CALL(
                meta::make_indices_c, impl_::max({tuple_util::size<GT_META_CALL(strides_type, Sids)>::value...}));

            template <class I>
            GT_META_DEFINE_ALIAS(get_stride_type,
                meta::id,
                (typename impl_::compressed<strides_map_t>::template composite<GT_META_CALL(
                        impl_::normalized_stride_type, (I, GT_META_CALL(strides_type, Sids)))...>));

            using ptr_t = impl_::composite_ptr<GT_META_CALL(ptr_type, Sids)...>;
            using strides_t = GT_META_CALL(meta::rename,
                (meta::ctor<tuple<>>::apply, GT_META_CALL(meta::transform, (get_stride_type, stride_indices_t))));

            using bounds_validator_t =
                typename impl_::compressed<bounds_validator_map_t>::template composite<bounds_validator_type<Sids>...>;
            using ptr_diff_t = typename impl_::compressed<strides_map_t>::template composite<ptr_diff_type<Sids>...>;

            friend constexpr GT_FUNCTION ptr_t sid_get_origin(composite &obj) {
                return tuple_util::host_device::transform(get_origin_f{}, obj.m_sids);
            }

            friend constexpr GT_FUNCTION strides_t sid_get_strides(composite const &obj) {
                return tuple_util::host_device::transform(impl_::converter_f{},
                    tuple_util::host_device::transpose(tuple_util::host_device::transform(
                        impl_::get_normalized_strides_f<stride_indices_t>{}, obj.m_sids)),
                    GT_META_CALL(meta::transform, (meta::lazy::id, strides_t)){});
            }

            friend constexpr GT_FUNCTION bounds_validator_t sid_get_bounds_validator(composite const &obj) {
                return tuple_util::host_device::transform(get_bounds_validator_f{}, obj.m_sids);
            }

            friend ptr_diff_t sid_get_ptr_diff(composite const &) { return {}; }
            friend GT_META_CALL(meta::dedup, strides_kinds_t) sid_get_strides_kind(composite const &) { return {}; }
            friend GT_META_CALL(meta::dedup, bounds_validator_kinds_t)
                sid_get_bounds_validator_kind(composite const &) {
                return {};
            }
            GT_TUPLE_UTIL_FORWARD_GETTER_TO_MEMBER(composite, m_sids);

          public:
            GT_TULPE_UTIL_FORWARD_CTORS_TO_MEMBER(composite, m_sids);
        };
    } // namespace sid
} // namespace gridtools
