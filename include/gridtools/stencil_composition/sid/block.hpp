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

#include "../../common/generic_metafunctions/for_each.hpp"
#include "../../common/host_device.hpp"
#include "../../common/hymap.hpp"
#include "../../common/tuple_util.hpp"
#include "../../meta/concat.hpp"
#include "../../meta/push_back.hpp"
#include "../../meta/rename.hpp"
#include "../../meta/st_contains.hpp"
#include "./concept.hpp"
#include "./delegate.hpp"

namespace gridtools {
    namespace sid {
        template <class Dim>
        struct blocked_dim {
            using type = blocked_dim<Dim>;
        };

        namespace block_impl_ {
            template <class Stride, class BlockSize>
            struct blocked_stride {
                Stride m_stride;
                BlockSize m_block_size;
            };

            template <class Ptr, class Stride, class BlockSize, class Offset>
            GT_FUNCTION auto sid_shift(Ptr &ptr, blocked_stride<Stride, BlockSize> const &stride, Offset const &offset)
                GT_AUTO_RETURN(shift(ptr, stride.m_stride, stride.m_block_size *offset));

            template <class Strides, class BlockMap>
            struct block_strides_f {
                Strides m_strides;
                BlockMap m_map;

                template <class Map, class Dim>
                using decay_at = decay_t<decltype(at_key<Dim>(std::declval<Map>()))>;

                template <class Dim>
                blocked_stride<decay_at<Strides, Dim>, decay_at<BlockMap, Dim>> operator()(blocked_dim<Dim>) const {
                    return {at_key<Dim>(m_strides), at_key<Dim>(m_map)};
                }

                template <class Dim>
                decay_at<Strides, Dim> operator()(Dim) const {
                    return at_key<Dim>(m_strides);
                }
            };

            template <class Strides, class BlockMap>
            using blocked_strides_keys = GT_META_CALL(meta::concat,
                (GT_META_CALL(get_keys, Strides),
                    GT_META_CALL(meta::transform, (blocked_dim, GT_META_CALL(get_keys, BlockMap)))));

            template <class Strides,
                class BlockMap,
                class Keys = blocked_strides_keys<Strides, BlockMap>,
                class KeysToKeys = GT_META_CALL(meta::rename, (Keys::template values, Keys))>
            auto block_strides(Strides const &strides, BlockMap const &block_map)
                GT_AUTO_RETURN(tuple_util::host_device::transform(
                    block_strides_f<Strides, BlockMap>{strides, block_map}, KeysToKeys{}));

            template <class Sid, class BlockMap>
            class blocked_sid : public delegate<Sid> {
                BlockMap m_block_map;
                using sid_strides_t = GT_META_CALL(strides_type, Sid);
                using sid_keys_t = GT_META_CALL(get_keys, sid_strides_t);
                using sid_vals_t = GT_META_CALL(
                    meta::rename, (meta::list, GT_META_CALL(tuple_util::traits::to_types, sid_strides_t)));
                using all_blocked_keys_t = GT_META_CALL(get_keys, BlockMap);

                template <class Dim>
                GT_META_DEFINE_ALIAS(original_dim, meta::st_contains, (sid_keys_t, Dim));
                using blocked_keys_t = GT_META_CALL(meta::filter, (original_dim, all_blocked_keys_t));

                template <class Map, class Dim>
                using decay_at = decay_t<decltype(at_key<Dim>(std::declval<Map>()))>;

                template <class Stride, class BlockSize>
                using blocked_stride_type =
                    conditional_t<std::is_integral<Stride>::value || concept_impl_::is_integral_constant<Stride>::value,
                        Stride,
                        blocked_stride<Stride, BlockSize>>;

                template <class Dim>
                using blocked_stride_at = blocked_stride_type<decay_at<sid_strides_t, Dim>, decay_at<BlockMap, Dim>>;

                using blocked_vals_t = GT_META_CALL(
                    meta::rename, (meta::list, GT_META_CALL(meta::transform, (blocked_stride_at, blocked_keys_t))));

                using new_keys_t = GT_META_CALL(
                    meta::concat, (sid_keys_t, GT_META_CALL(meta::transform, (blocked_dim, blocked_keys_t))));
                using new_vals_t = GT_META_CALL(meta::concat, (sid_vals_t, blocked_vals_t));

                using new_hymap_keys_t = typename GT_META_CALL(meta::rename, (hymap::keys, new_keys_t));

                using new_strides_t = GT_META_CALL(meta::rename, (new_hymap_keys_t::template values, new_vals_t));

                using foo = typename new_strides_t::foo;

              public:
                blocked_sid(Sid const &impl, BlockMap const &block_map) noexcept
                    : delegate<Sid>(impl), m_block_map(block_map) {}

                friend void sid_get_strides(blocked_sid &obj) { auto &&impl = obj.impl(); }
            };
        } // namespace block_impl_

        // template <class Sid, class BlockMap>
        // auto block(Sid const &s, BlockMap const &block_map)
        // GT_AUTO_RETURN((synthetic()
        //.set<property::origin>(get_origin(s))
        //.template set<property::strides>(block_impl_::block_strides(get_strides(s), block_map))
        //.template set<property::ptr_diff, ptr_diff_type<Sid>>()
        //.template set<property::strides_kind, strides_kind<Sid>>()));

        template <class Sid, class BlockMap>
        block_impl_::blocked_sid<Sid, BlockMap> block(Sid const &sid, BlockMap const &block_map) {
            return {sid, block_map};
        }
    } // namespace sid
} // namespace gridtools
