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
        vec_t m_lengths;
        vec_t m_strides;

      public:
        template <class Lengths, class Strides>
        storage_info_rt(Lengths const &lengths, Strides const &strides) {
            for (auto &&elem : lengths)
                m_lengths.push_back(elem);
            for (auto &&elem : strides)
                m_strides.push_back(elem);
        }

        vec_t const &lengths() const { return m_lengths; }
        vec_t const &strides() const { return m_strides; }
    };

    /*
     * @brief Construct a storage_info_rt from a storage_info
     */
    template <typename StorageInfo>
    storage_info_rt make_storage_info_rt(StorageInfo const &storage_info) {
        return {storage_info.lengths(), storage_info.strides()};
    }
    /**
     * @}
     */
} // namespace gridtools
