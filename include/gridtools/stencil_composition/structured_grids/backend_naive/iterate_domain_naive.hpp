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

#include <utility>

#include "../../../common/defs.hpp"
#include "../../../common/host_device.hpp"
#include "../../iterate_domain_fwd.hpp"
#include "../iterate_domain.hpp"

namespace gridtools {

    namespace iterate_domain_naive_impl_ {
        template <class StridesKind, bool IsTmp>
        struct get_index_offset_f;

        template <class StridesKind>
        struct get_index_offset_f<StridesKind, false> {
            template <class Stride, class Begin>
            int_t operator()(Stride const &stride, Begin const &begin) const {
                return stride.i * begin.i + stride.j * begin.j + stride.k * begin.k;
            }
        };

        template <class StorageInfo>
        struct get_index_offset_f<StorageInfo, true> {
            template <class Stride, class Begin>
            int_t operator()(Stride const &stride, Begin const &) const {
                return stride.i * StorageInfo::halo_t::template at<dim::i::value>() +
                       stride.j * StorageInfo::halo_t::template at<dim::j::value>();
            }
        };

        template <class StridesMap, class LocalDomain, class ArrayIndex>
        struct initialize_index_f {
            GT_STATIC_ASSERT((is_array_of<ArrayIndex, int>::value), GT_INTERNAL_ERROR);
            StridesMap const &m_strides_map;
            pos3<uint_t> const &m_begin;
            ArrayIndex &m_index_array;

            template <typename StridesKind>
            GT_FUNCTION void operator()() const {
                static constexpr auto index = _impl::get_index<StridesKind, LocalDomain>::value;
                GT_STATIC_ASSERT(index < ArrayIndex::size(), "Accessing an index out of bound in fusion tuple");
                static constexpr auto is_tmp =
                    meta::st_contains<typename LocalDomain::tmp_strides_kinds_t, StridesKind>::value;
                auto const &strides = host_device::at_key<StridesKind>(m_strides_map);
                m_index_array[index] =
                    get_index_offset_f<StridesKind, is_tmp>{}(make_pos3<int_t>(sid::get_stride<dim::i>(strides),
                                                                  sid::get_stride<dim::j>(strides),
                                                                  sid::get_stride<dim::k>(strides)),
                        m_begin);
            }
        };

        template <class LocalDomain, class StridesMap, class ArrayIndex>
        initialize_index_f<StridesMap, LocalDomain, ArrayIndex> initialize_index(
            StridesMap const &strides_map, pos3<uint_t> const &begin, ArrayIndex &index_array) {
            return {strides_map, begin, index_array};
        }
    } // namespace iterate_domain_naive_impl_

    /**
     * @brief iterate domain class for the X86 backend
     */
    template <class LocalDomain>
    class iterate_domain_naive {
        GT_STATIC_ASSERT(is_local_domain<LocalDomain>::value, GT_INTERNAL_ERROR);

        static const uint_t n_meta_storages = meta::length<typename LocalDomain::strides_kinds_t>::value;

        using array_index_t = array<int_t, n_meta_storages>;
        // ******************* members *******************
        LocalDomain const &local_domain;
        typename LocalDomain::ptr_map_t m_ptr_map;

        array_index_t m_index;
        // ******************* end of members *******************

      public:
        template <class Grid>
        iterate_domain_naive(LocalDomain const &local_domain, Grid const &grid)
            : local_domain(local_domain), m_ptr_map(local_domain.make_ptr_map()) {
            for_each_type<typename LocalDomain::strides_kinds_t>(
                iterate_domain_naive_impl_::initialize_index<LocalDomain>(
                    local_domain.m_strides_map, {grid.i_low_bound(), grid.j_low_bound(), grid.k_min()}, m_index));
        }

        template <class Dim, class Offset>
        void increment(Offset const &offset) {
            do_increment<Dim, LocalDomain>(offset, local_domain.m_strides_map, m_index);
        }

        template <class Offset = integral_constant<int_t, 1>>
        void increment_i(Offset const &offset = {}) {
            increment<dim::i>(offset);
        }
        template <class Offset = integral_constant<int_t, 1>>
        void increment_j(Offset const &offset = {}) {
            increment<dim::j>(offset);
        }
        template <class Offset = integral_constant<int_t, 1>>
        void increment_k(Offset const &offset = {}) {
            increment<dim::k>(offset);
        }

        template <class Arg, intent Intent, class Accessor, class Res = typename deref_type<Arg, Intent>::type>
        GT_FUNCTION Res deref(Accessor const &accessor) const {
            using data_t = typename Arg::data_store_t::data_t;
            using storage_info_t = typename Arg::data_store_t::storage_info_t;

            static constexpr auto storage_info_index =
                meta::st_position<typename LocalDomain::strides_kinds_t, storage_info_t>::value;

            auto pointer_offset = m_index[storage_info_index];
            sid::multi_shift(pointer_offset, host_device::at_key<storage_info_t>(local_domain.m_strides_map), accessor);

            return *(at_key<Arg>(m_ptr_map) + pointer_offset);
        }
    };

    template <class LocalDomain>
    struct is_iterate_domain<iterate_domain_naive<LocalDomain>> : std::true_type {};
} // namespace gridtools
