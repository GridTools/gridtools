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
#include <utility>

#include "../../common/array.hpp"
#include "../../common/defs.hpp"
#include "../../common/generic_metafunctions/accumulate.hpp"
#include "../../common/host_device.hpp"
#include "../../common/layout_map.hpp"
#include "alignment.hpp"
#include "halo.hpp"

namespace gridtools {

    /** \ingroup storage
     * @{
     */

    /**
     * @brief The storage info interface. This class contains methods that should be implemented by all storage_info
     * implementations.
     * @tparam Id unique ID that should be shared among all storage infos with the same dimensionality.
     * @tparam Layout information about the memory layout
     * @tparam Halo information about the halo sizes (by default no halo is set)
     * @tparam Alignment information about the alignment
     */
    template <uint_t Id,
        class Layout,
        class Halo = zero_halo<Layout::masked_length>,
        class Alignment = alignment<1>,
        class Dims = std::make_index_sequence<Layout::masked_length>>
    struct storage_info;

    template <uint_t Id, int... LayoutArgs, uint_t... Halos, uint_t Align, size_t... Dims>
    struct storage_info<Id, layout_map<LayoutArgs...>, halo<Halos...>, alignment<Align>, std::index_sequence<Dims...>> {
        static_assert(sizeof...(Halos) == sizeof...(LayoutArgs),
            GT_INTERNAL_ERROR_MSG("Halo size does not match number of dimensions"));

        using layout_t = layout_map<LayoutArgs...>;
        using alignment_t = alignment<Align>;

        static constexpr uint_t ndims = sizeof...(LayoutArgs);

      private:
        template <size_t>
        using index_type = uint_t;
        using array_t = array<uint_t, ndims>;

        static constexpr int max_layout_arg = constexpr_max(LayoutArgs...);

        array_t m_lengths;
        array_t m_strides;
        uint_t m_length = accumulate(logical_and(), true, m_lengths[Dims]...) ? index((m_lengths[Dims] - 1)...) + 1 : 0;

        static GT_FUNCTION GT_CONSTEXPR uint_t make_padded_length(int layout_arg, uint_t length) {
            if (layout_arg == -1)
                return 1;
            if (layout_arg == max_layout_arg)
                return (length + Align - 1) / Align * Align;
            return length;
        }

        static GT_FUNCTION GT_CONSTEXPR uint_t make_stride(int layout_arg, array_t const &padded_lengths) {
            if (layout_arg == -1)
                return 0;
            uint_t res = 1;
            for (int i = max_layout_arg; i != layout_arg; --i)
                res *= padded_lengths[layout_t::find(i)];
            return res;
        }

        GT_FUNCTION GT_CONSTEXPR bool is_stride_valid(int layout_arg, int dim) const {
            if (m_lengths[dim] == 0)
                return true;
            auto stride = m_strides[dim];
            if (layout_arg == -1)
                return stride == 0;
            if (layout_arg == max_layout_arg)
                return stride == 1 || stride % Align == 0;
            auto next_dim = layout_t::find(layout_arg + 1);
            if (stride % Align)
                return false;
            auto next_stride = m_strides[next_dim];
            if (stride % next_stride)
                return false;
            return m_lengths[next_dim] <= stride / next_stride;
        }

        GT_FUNCTION GT_CONSTEXPR storage_info(std::true_type, array_t const &lengths, array_t padded_lengths)
            : storage_info(lengths, {make_stride(LayoutArgs, padded_lengths)...}) {}

      public:
        constexpr static uint_t id = Id;

        storage_info() = default;

        /**
         * @brief storage info constructor. Additionally to initializing the members the halo
         * region is added to the corresponding dimensions and the alignment is applied.
         */
        GT_FUNCTION GT_CONSTEXPR storage_info(index_type<Dims>... lengths)
            : storage_info(std::true_type(), {lengths...}, {make_padded_length(LayoutArgs, lengths)...}) {}

        GT_FUNCTION GT_CONSTEXPR storage_info(array_t const &lengths, array_t const &strides)
            : m_lengths(lengths), m_strides(strides) {
            assert(accumulate(logical_and(), true, is_stride_valid(LayoutArgs, Dims)...));
        }

        /**
         * @brief Returns the array of total_lengths, the lengths including the halo points (the outer region)
         */
        GT_FUNCTION GT_CONSTEXPR auto const &lengths() const { return m_lengths; }

        /**
         * @brief return the array of (aligned) strides, see stride() for details.
         */
        GT_FUNCTION GT_CONSTEXPR auto const &strides() const { return m_strides; }

        /**
         * @brief member function to retrieve an offset (or index) when given offsets in I,J,K, etc.
         * E.g., index(1,2,3) --> 1*strideI + 2*strideJ + 3*strideK + initial_offset
         * @param idx given offsets
         * @return index
         */

        GT_FUNCTION GT_CONSTEXPR auto index(index_type<Dims>... indices) const {
            assert(accumulate(logical_and(), true, (indices < m_lengths[Dims])...));
            return accumulate(plus_functor(), 0, indices * m_strides[Dims]...);
        }

        /**
         * @brief member function to retrieve an offset (or index) when given an array of offsets in I,J,K, etc.
         * E.g., index(1,2,3) --> 1*strideI + 2*strideJ + 3*strideK + initial_offset
         * @param offsets given offset array
         * @return index
         */
        GT_FUNCTION GT_CONSTEXPR auto index(array<int, ndims> const &indices) const { return index(indices[Dims]...); }

        GT_FUNCTION GT_CONSTEXPR array_t indices(uint_t index) const {
            return {(LayoutArgs == -1 ? m_lengths[Dims] - 1
                                      : LayoutArgs ? index % m_strides[layout_t::find(LayoutArgs - 1)] / m_strides[Dims]
                                                   : index / m_strides[Dims])...};
        }

        GT_FUNCTION GT_CONSTEXPR auto length() const { return m_length; }

        /**
         * @brief function to check for equality of two given storage_infos
         * @param rhs right hand side storage info instance
         * @return true if the storage infos are equal, false otherwise
         */
        friend GT_FUNCTION GT_CONSTEXPR bool operator==(storage_info const &lhs, storage_info const &rhs) {
            return lhs.m_lengths == rhs.m_lengths && lhs.m_strides == rhs.m_strides;
        }

        friend GT_FUNCTION GT_CONSTEXPR bool operator!=(storage_info const &lhs, storage_info const &rhs) {
            return !(lhs == rhs);
        }
    };

    template <class>
    struct is_storage_info : std::false_type {};

    template <uint_t Id, class Layout, class Halo, class Alignment, class Indices>
    struct is_storage_info<storage_info<Id, Layout, Halo, Alignment, Indices>> : std::true_type {};

    /**
     * @}
     */
} // namespace gridtools
