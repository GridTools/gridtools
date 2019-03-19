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

#include <cassert>

#include "../../common/binops.hpp"
#include "../../common/defs.hpp"
#include "../../common/generic_metafunctions/for_each.hpp"
#include "../../common/generic_metafunctions/utility.hpp"
#include "../../common/host_device.hpp"
#include "../../common/hymap.hpp"
#include "../../common/tuple.hpp"
#include "../../common/tuple_util.hpp"
#include "../../meta.hpp"
#include "concept.hpp"

namespace gridtools {
    namespace sid {
        namespace composite {
            namespace impl_ {
                struct deref_f {
                    template <class T>
                    constexpr GT_FUNCTION auto operator()(T const &obj) const GT_AUTO_RETURN(*obj);
                };

                struct call_f {
                    template <class T>
                    constexpr GT_FUNCTION auto operator()(T const &obj) const -> decay_t<decltype(obj())> {
                        return obj();
                    }
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
                 *   - the first overload can be chosen only if `lhs == rhs` defined [SFINAE: `decltype(lhs == rhs)` is
                 * a part of the signature]
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
                            (conjunction<
                                std::is_same<GT_META_CALL(tuple_util::element, (SecondaryIndices::value, Args)),
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

                template <class Key, class Strides, class I = meta::st_position<GT_META_CALL(get_keys, Strides), Key>>
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

                template <class Keys>
                struct normalize_strides_f;

                template <template <class...> class L, class... Keys>
                struct normalize_strides_f<L<Keys...>> {
                    template <class Sid, class Strides = GT_META_CALL(strides_type, Sid)>
                    constexpr tuple<GT_META_CALL(normalized_stride_type, (Keys, decay_t<Strides>))...> operator()(
                        Sid const &sid) const {
                        return {get_stride<Keys>(get_strides(sid))...};
                    }
                };
            } // namespace impl_

            /**
             *  This class models both `SID` and `hymap` concepts at the same time
             *
             *  All derived types of the `composite` are `hymap`'s of the correspondent types of the original ones.
             *  Example:
             *    say you have two sids: `s1` and `s2` of types `S1` and `S2`
             *    you can compose them: `composite::keys<a, b>::values<S1, S2> c = {s1, s2};`
             *    now `c` is a `SID` as well, you can call `get_origin(c)`, `get_strides(c)` etc.
             *    you have an access to `s1` and `s2` via `get_key`: `get_key<a>(c)` is the same as `s1`
             *
             *  The way how the composite deals with strides is easy to illustrate with example:
             *  Say the sid `s1` has strides {dim_i:3, dim_j:15} (here I use pseudo code to experess the map)
             *  and `s2` has strides {dim_k:4, dim_j:12}.
             *  We create a composite `c`: `composite::keys<a, b>::values<S1, S2> c = {s1, s2}`
             *  Now `c` has strides : {dim_i:{a:3, b:0}, dim_j:{a:15, b:12}, dim_k:{a:0, b:4}}
             *
             * Internally `composite` utilizes the fact that the strides of the same kind are always the same.
             * This  means that the composite stride doesn't hold duplicate strides.
             *
             */
            template <class... Keys>
            class keys {

                template <class... Ptrs>
                struct composite_ptr {
                    GT_STATIC_ASSERT(sizeof...(Keys) == sizeof...(Ptrs), GT_INTERNAL_ERROR);

                    typename hymap::keys<Keys...>::template values<Ptrs...> m_vals;
                    GT_TUPLE_UTIL_FORWARD_GETTER_TO_MEMBER(composite_ptr, m_vals);
                    GT_TUPLE_UTIL_FORWARD_CTORS_TO_MEMBER(composite_ptr, m_vals);
                    constexpr GT_FUNCTION auto operator*() const
                        GT_AUTO_RETURN(tuple_util::host_device::transform(impl_::deref_f{}, m_vals));

                    friend keys hymap_get_keys(composite_ptr const &) { return {}; }
                };

                template <class... PtrHolders>
                struct composite_ptr_holder {
                    GT_STATIC_ASSERT(sizeof...(Keys) == sizeof...(PtrHolders), GT_INTERNAL_ERROR);

                    typename hymap::keys<Keys...>::template values<PtrHolders...> m_vals;
                    GT_TUPLE_UTIL_FORWARD_GETTER_TO_MEMBER(composite_ptr_holder, m_vals);
                    GT_TUPLE_UTIL_FORWARD_CTORS_TO_MEMBER(composite_ptr_holder, m_vals);

                    constexpr GT_FUNCTION auto operator()() const
                        GT_AUTO_RETURN(tuple_util::host_device::convert_to<composite_ptr>(
                            tuple_util::host_device::transform(impl_::call_f{}, m_vals)));

                    friend keys hymap_get_keys(composite_ptr_holder const &) { return {}; }
                };

#if GT_BROKEN_TEMPLATE_ALIASES
              public:
#endif
                /**
                 *  Implements strides and ptr_diffs compression based on skipping the objects of the
                 *  same kind.
                 *
                 *  `composite` objects pretend to be a tuples of `uncompressed` types (aka external tuple), but
                 * internally they store a tuple of `compressed` types (aka internal tuple).
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

                    using generators_t = GT_META_CALL(meta::transform, (impl_::item_generator, inversed_map_t));

                    template <size_t I,
                        class T,
                        class Item = GT_META_CALL(meta::mp_find, (Map, std::integral_constant<size_t, I>)),
                        class Pos = GT_META_CALL(meta::second, Item)>
                    static constexpr GT_FUNCTION auto get(T &&obj)
                        GT_AUTO_RETURN(tuple_util::host_device::get<Pos::value>(const_expr::forward<T>(obj).m_vals));

                    template <class... Ts>
                    struct composite_entity {
                        GT_STATIC_ASSERT(sizeof...(Keys) == sizeof...(Ts), GT_INTERNAL_ERROR);

                        template <class I>
                        GT_META_DEFINE_ALIAS(get_compressed_type, meta::at, (meta::list<Ts...>, I));

                        using vals_t = GT_META_CALL(meta::transform, (get_compressed_type, primary_indices_t));

                        vals_t m_vals;

                        template <class... Args, enable_if_t<sizeof...(Args) == sizeof...(Ts), int> = 0>
                        constexpr composite_entity(Args &&... args) noexcept
                            : composite_entity(tuple<Args &&...>{std::forward<Args &&>(args)...}) {}

                        template <template <class...> class L,
                            class... Args,
                            enable_if_t<sizeof...(Args) == sizeof...(Ts), int> = 0>
                        constexpr composite_entity(L<Args...> &&tup) noexcept
                            : m_vals{tuple_util::generate<generators_t, vals_t>(std::move(tup))} {}

                        GT_DECLARE_DEFAULT_EMPTY_CTOR(composite_entity);
                        composite_entity(composite_entity const &) = default;
                        composite_entity(composite_entity &&) noexcept = default;
                        composite_entity &operator=(composite_entity const &) = default;
                        composite_entity &operator=(composite_entity &&) noexcept = default;

                        friend compressed tuple_getter(composite_entity const &) { return {}; }

                        template <class... Ptrs>
                        friend constexpr GT_FUNCTION composite_ptr<Ptrs...> operator+(
                            composite_ptr<Ptrs...> const &lhs, composite_entity const &rhs) {
                            return tuple_util::host_device::transform(binop::sum{}, lhs, rhs);
                        }

                        template <class... Ptrs, class Offset>
                        friend GT_FUNCTION void sid_shift(composite_ptr<Ptrs...> &ptr,
                            composite_entity const &stride,
                            Offset const &GT_RESTRICT offset) {
                            impl_::composite_shift_impl(ptr.m_vals, stride, offset);
                        }

                        friend keys hymap_get_keys(composite_entity const &) { return {}; }
                    };

                    template <class... PtrDiffs, class... Strides, class Offset>
                    friend GT_FUNCTION void sid_shift(composite_entity<PtrDiffs...> &GT_RESTRICT ptr_diff,
                        composite_entity<Strides...> const &GT_RESTRICT stride,
                        Offset const &GT_RESTRICT offset) {
                        impl_::composite_shift_impl(ptr_diff.m_vals, stride.m_vals, offset);
                    }

                    struct convert_f {
                        template <template <class...> class L, class... Ts>
                        constexpr composite_entity<remove_reference_t<Ts>...> operator()(L<Ts...> &&tup) const {
                            return {std::move(tup)};
                        }
                    };
                };

              public:
                template <class... Sids>
                class values {
                    GT_STATIC_ASSERT(sizeof...(Keys) == sizeof...(Sids), GT_INTERNAL_ERROR);
                    GT_STATIC_ASSERT(conjunction<is_sid<Sids>...>::value, GT_INTERNAL_ERROR);

                    typename hymap::keys<Keys...>::template values<Sids...> m_sids;

                    // Extracted lists of raw kinds (uncompresed)
                    using strides_kinds_t = meta::list<GT_META_CALL(strides_kind, Sids)...>;

                    // The index map that is needed to build compressed composite objects
                    using map_t = GT_META_CALL(impl_::make_index_map, strides_kinds_t);
                    using compressed_t = compressed<map_t>;

                    template <class... Ts>
                    GT_META_DEFINE_ALIAS(compress, meta::id, (typename compressed_t::template composite_entity<Ts...>));

                    using stride_keys_t = GT_META_CALL(meta::dedup,
                        GT_META_CALL(meta::concat, GT_META_CALL(get_keys, GT_META_CALL(strides_type, Sids))...));

#if defined(__CUDACC_VER_MAJOR__) && __CUDACC_VER_MAJOR__ == 9 && __CUDACC_VER_MINOR__ < 2
                    struct stride_hymap_keys {
                        using type = GT_META_CALL(meta::rename, (hymap::keys, stride_keys_t));
                    };
                    using stride_hymap_keys_t = typename stride_hymap_keys::type;
#else
                    using stride_hymap_keys_t = GT_META_CALL(meta::rename, (hymap::keys, stride_keys_t));
#endif

                    template <class... Values>
                    GT_META_DEFINE_ALIAS(
                        stride_hymap_ctor, meta::id, typename stride_hymap_keys_t::template values<Values...>);

                    // A helper for generating strides_t
                    // It is a meta function from the stride key to the stride type
                    template <class Key>
                    GT_META_DEFINE_ALIAS(get_stride_type,
                        compress,
                        (GT_META_CALL(
                            impl_::normalized_stride_type, (Key, decay_t < GT_META_CALL(strides_type, Sids)) >)...));

                    // all `SID` types are here
                    using ptr_holder_t = composite_ptr_holder<GT_META_CALL(ptr_holder_type, Sids)...>;
                    using ptr_t = composite_ptr<GT_META_CALL(ptr_type, Sids)...>;
                    using strides_t = GT_META_CALL(meta::rename,
                        (stride_hymap_ctor, GT_META_CALL(meta::transform, (get_stride_type, stride_keys_t))));
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

                    friend GT_SID_COMPOSITE_CONSTEXPR ptr_holder_t sid_get_origin(values &obj) {
                        return tuple_util::transform(get_origin_f{}, obj.m_sids);
                    }

                    friend GT_SID_COMPOSITE_CONSTEXPR strides_t sid_get_strides(values const &obj) {
                        return tuple_util::convert_to<stride_hymap_keys_t::template values>(
                            tuple_util::transform(typename compressed_t::convert_f{},
                                tuple_util::transpose(
                                    tuple_util::transform(impl_::normalize_strides_f<stride_keys_t>{}, obj.m_sids))));
                    }

#undef GT_SID_COMPOSITE_CONSTEXPR

                    friend ptr_diff_t sid_get_ptr_diff(values const &) { return {}; }

                    friend GT_META_CALL(meta::dedup, strides_kinds_t) sid_get_strides_kind(values const &) {
                        return {};
                    }

                    // Here the `tuple_like` concept is modeled

                    GT_TUPLE_UTIL_FORWARD_GETTER_TO_MEMBER(values, m_sids);
                    GT_TUPLE_UTIL_FORWARD_CTORS_TO_MEMBER(values, m_sids);

                    // hymap concept
                    friend keys hymap_get_keys(values const &) { return {}; }
                };
            };
        } // namespace composite
    }     // namespace sid
} // namespace gridtools
