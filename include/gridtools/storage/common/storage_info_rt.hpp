/*
 * GridTools Libraries
 *
 * Copyright (c) 2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include <vector>

#include "../../common/defs.hpp"

namespace gridtools {
    /** \ingroup storage
     * @{
     */

    /*
     * @brief A storage info runtime object, providing dimensions and strides but without type information of the
     * storage info. Can be used in interfaces where no strict type information is available, i.e. in the interface to
     * Fortran.
     */
    class storage_info_rt {
        using vec_t = std::vector<uint_t>;
        vec_t m_total_lengths;
        vec_t m_padded_lengths;
        vec_t m_strides;

      public:
        template <class TotalLengths, class PaddedLengths, class Strides>
        storage_info_rt(
            TotalLengths const &total_lengths, PaddedLengths const &padded_lengths, Strides const &strides) {
            for (auto &&elem : total_lengths)
                m_total_lengths.push_back(elem);
            for (auto &&elem : padded_lengths)
                m_padded_lengths.push_back(elem);
            for (auto &&elem : strides)
                m_strides.push_back(elem);
        }

        const vec_t &total_lengths() const { return m_total_lengths; }
        const vec_t &padded_lengths() const { return m_padded_lengths; }
        const vec_t &strides() const { return m_strides; }
    };

    /*
     * @brief Construct a storage_info_rt from a storage_info
     */
    template <typename StorageInfo>
    storage_info_rt make_storage_info_rt(StorageInfo const &storage_info) {
        return {storage_info.total_lengths(), storage_info.padded_lengths(), storage_info.strides()};
    }
    /**
     * @}
     */
} // namespace gridtools
