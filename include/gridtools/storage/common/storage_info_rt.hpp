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
