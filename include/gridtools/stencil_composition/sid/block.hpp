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

#include <functional>
#include <type_traits>
#include <utility>

#include "../../common/defs.hpp"
#include "../../common/generic_metafunctions/for_each.hpp"
#include "../../common/host_device.hpp"
#include "../../common/hymap.hpp"
#include "../../common/tuple.hpp"
#include "../../common/tuple_util.hpp"
#include "../../meta.hpp"
#include "./concept.hpp"
#include "./delegate.hpp"

namespace gridtools {
    namespace sid {
        template <class Dim>
        struct blocked_dim {
            using type = blocked_dim;
        };

        namespace block_impl_ {
            template <class Stride, class BlockSize>
            struct blocked_stride : tuple<Stride, BlockSize> {
                using tuple<Stride, BlockSize>::tuple;
            };

            template <class Ptr, class Stride, class BlockSize, class Offset>
            GT_FUNCTION auto sid_shift(Ptr &ptr, blocked_stride<Stride, BlockSize> const &stride, Offset const &offset)
                GT_AUTO_RETURN(shift(
                    ptr, tuple_util::host_device::get<0>(stride), tuple_util::host_device::get<1>(stride) * offset));

            template <class Stride, class BlockSize, enable_if_t<!std::is_integral<Stride>::value, int> = 0>
            blocked_stride<Stride, BlockSize> block_stride(Stride const &stride, BlockSize const &block_size) {
                return {stride, block_size};
            }

            template <class Stride, class BlockSize, enable_if_t<std::is_integral<Stride>::value, int> = 0>
            auto block_stride(Stride const &stride, BlockSize const &block_size) GT_AUTO_RETURN(stride *block_size);

            template <class T, T V, class BlockSize>
            auto block_stride(std::integral_constant<T, V> const &, BlockSize const &block_size)
                GT_AUTO_RETURN(V *block_size);

            template <class Stride, class BlockSize>
            using blocked_stride_type = decltype(block_stride(std::declval<Stride>(), std::declval<BlockSize>()));

            template <class Dim>
            struct generate_original_strides_f {
                using type = generate_original_strides_f;

                template <class Strides, class BlockMap>
                auto operator()(Strides const &strides, BlockMap const &) const GT_AUTO_RETURN(at_key<Dim>(strides));
            };

            template <class Dim>
            struct generate_blocked_strides_f {
                using type = generate_blocked_strides_f;

                template <class Strides, class BlockMap>
                auto operator()(Strides const &strides, BlockMap const &block_map) const
                    GT_AUTO_RETURN(block_stride(at_key<Dim>(strides), at_key<Dim>(block_map)));
            };

            template <class Sid, class BlockMap>
            class blocked_sid : public delegate<Sid> {
                BlockMap m_block_map;

                using strides_map_t = GT_META_CALL(hymap::to_meta_map, GT_META_CALL(strides_type, Sid));
                using strides_dims_t = GT_META_CALL(meta::transform, (meta::first, strides_map_t));

                template <class MapEntry, class Dim = GT_META_CALL(meta::first, MapEntry)>
                GT_META_DEFINE_ALIAS(is_strides_dim, meta::st_contains, (strides_dims_t, Dim));

                using block_map_t = GT_META_CALL(
                    meta::filter, (is_strides_dim, GT_META_CALL(hymap::to_meta_map, BlockMap)));
                using block_dims_t = GT_META_CALL(meta::transform, (meta::first, block_map_t));

                template <class MapEntry,
                    class Dim = GT_META_CALL(meta::first, MapEntry),
                    class BlockSize = GT_META_CALL(meta::second, MapEntry),
                    class Stride = GT_META_CALL(meta::second, (GT_META_CALL(meta::mp_find, (strides_map_t, Dim))))>
                GT_META_DEFINE_ALIAS(block, meta::list, (blocked_dim<Dim>, blocked_stride_type<Stride, BlockSize>));

                using blocked_strides_map_t = GT_META_CALL(meta::transform, (block, block_map_t));

                using blocked_dims_t = GT_META_CALL(meta::transform, (meta::first, blocked_strides_map_t));
                GT_STATIC_ASSERT(
                    meta::is_empty<GT_META_CALL(meta::filter,
                        (meta::curry<meta::st_contains, strides_dims_t>::template apply, blocked_dims_t))>::value,
                    GT_INTERNAL_ERROR_MSG("tried to block already blocked dimension"));

                using strides_t = GT_META_CALL(hymap::from_meta_map,
                    (GT_META_CALL(meta::concat, (strides_map_t, GT_META_CALL(meta::transform, (block, block_map_t))))));

                using original_generators_t = GT_META_CALL(
                    meta::transform, (generate_original_strides_f, strides_dims_t));
                using blocked_generators_t = GT_META_CALL(meta::transform, (generate_blocked_strides_f, block_dims_t));
                using generators_t = GT_META_CALL(meta::concat, (original_generators_t, blocked_generators_t));

              public:
                template <class SidT, class BlockMapT>
                blocked_sid(SidT &&impl, BlockMapT &&block_map) noexcept
                    : delegate<Sid>(std::forward<SidT>(impl)), m_block_map(std::forward<BlockMapT>(block_map)) {}

                friend strides_t sid_get_strides(blocked_sid const &obj) {
                    return tuple_util::host_device::generate<generators_t, strides_t>(
                        get_strides(obj.impl()), obj.m_block_map);
                }
            };

            template <class Sid,
                class BlockMap,
                class SidDims = GT_META_CALL(get_keys, GT_META_CALL(strides_type, decay_t<Sid>)),
                class BlockMapDims = GT_META_CALL(get_keys, decay_t<BlockMap>)>
            using no_common_dims = meta::is_empty<GT_META_CALL(
                meta::filter, (meta::curry<meta::st_contains, SidDims>::template apply, BlockMapDims))>;

        } // namespace block_impl_

        template <class Sid, class BlockMap, enable_if_t<block_impl_::no_common_dims<Sid, BlockMap>::value, int> = 0>
        auto block(Sid &&sid, BlockMap &&) GT_AUTO_RETURN(std::forward<Sid>(sid));

        template <class Sid, class BlockMap, enable_if_t<!block_impl_::no_common_dims<Sid, BlockMap>::value, int> = 0>
        block_impl_::blocked_sid<decay_t<Sid>, decay_t<BlockMap>> block(Sid &&sid, BlockMap &&block_map) {
            return {std::forward<Sid>(sid), std::forward<BlockMap>(block_map)};
        }

        template <class Sid, class BlockMap>
        auto block(std::reference_wrapper<Sid> const &sid, BlockMap &&block_map)
            GT_AUTO_RETURN(block(sid.get(), std::forward<BlockMap>(block_map)));
    } // namespace sid
} // namespace gridtools
