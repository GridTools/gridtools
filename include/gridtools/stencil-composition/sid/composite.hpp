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
#include "../../common/generic_metafunctions/for_each.hpp"
#include "../../common/generic_metafunctions/utility.hpp"
#include "../../common/host_device.hpp"
#include "../../common/tuple.hpp"
#include "../../common/tuple_util.hpp"
#include "../../meta.hpp"
#include "concept.hpp"

namespace gridtools {
    namespace sid {
        namespace composite_impl_ {
            struct deref_f {
                template <class T>
                constexpr GT_FUNCTION auto operator()(T const &obj) const GT_AUTO_RETURN(*obj);
            };

            template <class... Ptrs>
            struct composite_ptr {
                tuple<Ptrs...> m_vals;
                GT_TUPLE_UTIL_FORWARD_GETTER_TO_MEMBER(composite_ptr, m_vals);
                GT_TUPLE_UTIL_FORWARD_CTORS_TO_MEMBER(composite_ptr, m_vals);
                constexpr GT_FUNCTION auto operator*() const
                    GT_AUTO_RETURN(tuple_util::host_device::transform(deref_f{}, m_vals));
            };

            GT_META_LAZY_NAMESPACE {
                template <class State, class Kind>
                struct make_map_helper {
                    using map_t = GT_META_CALL(meta::first, State);
                    using kinds_t = GT_META_CALL(meta::second, State);
                    using pos_t = typename meta::length<map_t>::type;
                    using kind_pos_t = typename meta::find<kinds_t, Kind>::type;
                    using new_map_t = GT_META_CALL(meta::push_back, (map_t, tuple<pos_t, kind_pos_t>));
                    using new_kinds_t = GT_META_CALL(meta::if_c,
                        (kind_pos_t::value == meta::length<kinds_t>::value,
                            GT_META_CALL(meta::push_back, (kinds_t, Kind)),
                            kinds_t));
                    using type = meta::list<new_map_t, new_kinds_t>;
                };
            }
            GT_META_DELEGATE_TO_LAZY(make_map_helper, (class State, class Kind), (State, Kind));

            template <class Kinds>
            GT_META_DEFINE_ALIAS(make_index_map,
                meta::first,
                (GT_META_CALL(meta::lfold, (make_map_helper, meta::list<tuple<>, meta::list<>>, Kinds))));

            /**
             *  `maybe_equal(lhs, rhs)` is a functional equivalent of the following pseudo code:
             *   `<no_equal_operator_exists> || lhs == rhs;`
             *
             *   It is implemented as following:
             *   - the first overload can be chosen only if `lhs == rhs` defined [SFINAE: `decltype(lhs == rhs)` is a
             *     part of the signature]
             *   - the second overload can be chosen only if the first failed.
             */
            template <class T>
            auto maybe_equal(T const &lhs, T const &rhs) -> decltype(lhs == rhs) {
                return lhs == rhs;
            }

            GT_FORCE_INLINE bool maybe_equal(...) { return true; }

            template <class PrimaryValue, class Tup>
            bool are_secondaries_equal_to_primary(PrimaryValue const &, Tup const &) {
                return true;
            }

            template <class SecondaryIndex, class... SecondaryIndices, class PrimaryValue, class Tup>
            bool are_secondaries_equal_to_primary(PrimaryValue const &primary_value, Tup const &tup) {
                return are_secondaries_equal_to_primary<SecondaryIndices...>(primary_value, tup) &&
                       maybe_equal(tuple_util::get<SecondaryIndex::value>(tup), primary_value);
            }

            template <class>
            struct item_generator;

            template <template <class...> class L, class Key, class PrimaryIndex, class... SecondaryIndices>
            struct item_generator<L<Key, PrimaryIndex, SecondaryIndices...>> {
                using type = item_generator;

                template <class Args, class Res = GT_META_CALL(tuple_util::element, (PrimaryIndex::value, Args))>
                Res const &operator()(Args const &args) const noexcept {
                    GT_STATIC_ASSERT(
                        (conjunction<std::is_same<GT_META_CALL(tuple_util::element, (SecondaryIndices::value, Args)),
                                Res>...>::value),
                        GT_INTERNAL_ERROR);
                    assert((are_secondaries_equal_to_primary<SecondaryIndices...>(
                        tuple_util::get<PrimaryIndex::value>(args), args)));
                    return tuple_util::get<PrimaryIndex::value>(args);
                }
            };

            template <class ObjTup, class StrideTup, class Offset>
            struct shift_t {
                ObjTup &GT_RESTRICT m_obj_tup;
                StrideTup const &GT_RESTRICT m_stride_tup;
                Offset const &GT_RESTRICT m_offset;

                template <class I>
                GT_FUNCTION void operator()() const {
                    shift(tuple_util::host_device::get<I::value>(m_obj_tup),
                        tuple_util::host_device::get<I::value>(m_stride_tup),
                        m_offset);
                }
            };

            template <class ObjTup, class StrideTup, class Offset>
            GT_FUNCTION void composite_shift_impl(ObjTup &GT_RESTRICT obj_tup,
                StrideTup const &GT_RESTRICT stride_tup,
                Offset const &GT_RESTRICT offset) {
                static constexpr size_t size = tuple_util::size<ObjTup>::value;
                GT_STATIC_ASSERT(tuple_util::size<StrideTup>::value == size, GT_INTERNAL_ERROR);
                host_device::for_each_type<GT_META_CALL(meta::make_indices_c, size)>(
                    shift_t<ObjTup, StrideTup, Offset>{obj_tup, stride_tup, offset});
            }

            /**
             *  Implements strides and ptr_diffs compression based on skipping the objects of the
             *  same kind.
             *
             *  `composite` objects pretend to be a tuples of `uncompressed` types (aka external tuple), but internally
             *  they store a tuple of `compressed` types (aka internal tuple).
             *
             *  @tparam Map - compile time map from the index in external tuple to the index of internal tuple.
             *                `compile time map` here is a tuple of tuples: `tuple<tuple<Key, Value>...>`
             */
            template <class Map>
            struct compressed {
                /**
                 *  Inverse map is like that: tuple<tuple<Value, Keys...>...>, where values and keys are taken
                 *  from the source.
                 *
                 *  In the concrete case of the map of indices it would be:
                 *  tuple<tuple<CompressedIndex, UncompressedIndices...>>
                 *
                 *  Note that it could be several UncompressedIndices per one CompressedIndex
                 *
                 */
                using inversed_map_t = GT_META_CALL(meta::mp_inverse, Map);
                /**
                 *  A tuple of the first of UncompressedIndices per each CompressedIndex
                 */
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

                    using vals_t = GT_META_CALL(meta::transform, (get_compressed_type, primary_indices_t));

                    vals_t m_vals;

                    template <class... Args, enable_if_t<sizeof...(Args) == sizeof...(Ts), int> = 0>
                    constexpr composite(Args &&... args) noexcept
                        : composite(tuple<Args &&...>{std::forward<Args &&>(args)...}) {}

                    template <class... Args, enable_if_t<sizeof...(Args) == sizeof...(Ts), int> = 0>
                    constexpr composite(tuple<Args...> &&tup) noexcept
                        : m_vals{tuple_util::generate<generators_t, vals_t>(std::move(tup))} {}

                    GT_DECLARE_DEFAULT_EMPTY_CTOR(composite);
                    composite(composite const &) = default;
                    composite(composite &&) noexcept = default;
                    composite &operator=(composite const &) = default;
                    composite &operator=(composite &&) noexcept = default;

                    friend compressed tuple_getter(composite const &) { return {}; }

                    template <class... Ptrs>
                    friend constexpr GT_FUNCTION composite_ptr<Ptrs...> operator+(
                        composite_ptr<Ptrs...> const &lhs, composite const &rhs) {
                        return tuple_util::host_device::transform(binop::sum{}, lhs, rhs);
                    }

                    template <class... Ptrs, class Offset>
                    friend GT_FUNCTION void sid_shift(
                        composite_ptr<Ptrs...> &ptr, composite const &stride, Offset const &GT_RESTRICT offset) {
                        composite_shift_impl(ptr.m_vals, stride, offset);
                    }
                };

                template <class... PtrDiffs, class... Strides, class Offset>
                friend GT_FUNCTION void sid_shift(composite<PtrDiffs...> &GT_RESTRICT ptr_diff,
                    composite<Strides...> const &GT_RESTRICT stride,
                    Offset const &GT_RESTRICT offset) {
                    composite_shift_impl(ptr_diff.m_vals, stride.m_vals, offset);
                }

                struct convert_f {
                    template <class... Ts>
                    constexpr composite<remove_reference_t<Ts>...> operator()(tuple<Ts...> &&tup) const {
                        return {std::move(tup)};
                    }
                };
            };

            constexpr size_t max_impl(size_t acc, size_t const *cur, size_t const *last) {
                return cur == last ? acc : max_impl(*cur > acc ? *cur : acc, cur + 1, last);
            }

            template <size_t... Vals>
            struct max {
                static constexpr size_t values[] = {Vals...};
                static constexpr size_t value = max_impl(0, values, values + sizeof...(Vals));
            };

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

            template <class StrideIndices>
            struct normalize_strides_f;

            template <template <class...> class L, class... Is>
            struct normalize_strides_f<L<Is...>> {
                template <class Sid, class Strides = GT_META_CALL(strides_type, Sid)>
                constexpr tuple<GT_META_CALL(normalized_stride_type, (Is, decay_t<Strides>))...> operator()(
                    Sid const &sid) const {
                    return {get_stride<Is::value>(get_strides(sid))...};
                }
            };
        } // namespace composite_impl_

        /**
         *  This class models both `SID` and `tuple_like` concepts at the same time
         *
         *  All derived types of the `composite` are `tuple_like`s of the correspondent types of the original ones.
         *  Example:
         *    say you have two sids: `s1` and `s2` of types `S1` and `S2`
         *    you can compose them: `comoposite<S1, S2> c = {s1, s2};`
         *    now `c` is a `SID` as well, you can call `get_origin(c)`, `get_strides(c)` etc.
         *    you have an access to `s1` and `s2` via `tuple_util::get`: `tuple_util::get<0>(c)` is the same as `s1`
         *
         *    When composing strides together the maximum strides size is calculated and for the original strides are
         *    expanded to that maximum with `integral_constant<int_t, 0>`.
         *    Internaly `composite` utilizes the fact that the strides of the same kind are always the same. This  means
         *    that the composite stride doesn't hold duplicate strides.
         *
         *  @tparam Sids - all of them should model `SID` concept
         */
        template <class... Sids>
        class composite {
            GT_STATIC_ASSERT(conjunction<is_sid<Sids>...>::value, GT_INTERNAL_ERROR);

            tuple<Sids...> m_sids;

            // Extracted lists of raw kinds (uncompresed)
            using strides_kinds_t = meta::list<GT_META_CALL(strides_kind, Sids)...>;

            // The index map that is needed to build compressed composite objects
            using map_t = GT_META_CALL(composite_impl_::make_index_map, strides_kinds_t);
            using compressed_t = composite_impl_::compressed<map_t>;

            template <class... Ts>
            GT_META_DEFINE_ALIAS(compress, meta::id, (typename compressed_t::template composite<Ts...>));

            // A helper for generating strides_t
            // It is a tuple containing indices of the strides to generate
            using stride_indices_t = GT_META_CALL(meta::make_indices_c,
                (composite_impl_::max<tuple_util::size<GT_META_CALL(strides_type, Sids)>::value...>::value, tuple));

            // A helper for generating strides_t
            // It is a metafuntion from the stride index to the stride type
            template <class I>
            GT_META_DEFINE_ALIAS(get_stride_type,
                compress,
                (GT_META_CALL(
                    composite_impl_::normalized_stride_type, (I, decay_t < GT_META_CALL(strides_type, Sids)) >)...));

            // all `SID` types are here
            using ptr_t = composite_impl_::composite_ptr<GT_META_CALL(ptr_type, Sids)...>;
            using strides_t = GT_META_CALL(meta::transform, (get_stride_type, stride_indices_t));
            using ptr_diff_t = GT_META_CALL(compress, (GT_META_CALL(ptr_diff_type, Sids)...));

          public:
            // Here the `SID` concept is modeled

#if defined(__CUDACC_VER_MAJOR__) && __CUDACC_VER_MAJOR__ < 9
            // Shame on you CUDA 8!!!
            // Why on the Earth a composition of `constexpr` functions could fail to be `constexpr`?
#define GT_SID_COMPOSITE_CONSTEXPR
#else
#define GT_SID_COMPOSITE_CONSTEXPR constexpr
#endif

            friend GT_SID_COMPOSITE_CONSTEXPR ptr_t sid_get_origin(composite &obj) {
                return tuple_util::transform(get_origin_f{}, obj.m_sids);
            }

            friend GT_SID_COMPOSITE_CONSTEXPR strides_t sid_get_strides(composite const &obj) {
                return tuple_util::transform(typename compressed_t::convert_f{},
                    tuple_util::transpose(
                        tuple_util::transform(composite_impl_::normalize_strides_f<stride_indices_t>{}, obj.m_sids)));
            }

#undef GT_SID_COMPOSITE_CONSTEXPR

            friend ptr_diff_t sid_get_ptr_diff(composite const &) { return {}; }

            friend GT_META_CALL(meta::dedup, strides_kinds_t) sid_get_strides_kind(composite const &) { return {}; }

            // Here the `tuple_like` concept is modeled

            GT_TUPLE_UTIL_FORWARD_GETTER_TO_MEMBER(composite, m_sids);
            GT_TUPLE_UTIL_FORWARD_CTORS_TO_MEMBER(composite, m_sids);
        };
    } // namespace sid
} // namespace gridtools
