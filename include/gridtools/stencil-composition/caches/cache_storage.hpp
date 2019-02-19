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

#include <type_traits>

#include "../../common/defs.hpp"
#include "../../common/gt_assert.hpp"
#include "../../common/host_device.hpp"
#include "../../meta/type_traits.hpp"
#include "../execution_types.hpp"
#include "../offset_computation.hpp"

namespace gridtools {

#ifdef GT_STRUCTURED_GRIDS
    template <class T, int_t ISize, int_t JSize, int_t IZero, int_t JZero>
    class ij_cache_storage {
        GT_STATIC_ASSERT(ISize > 0, GT_INTERNAL_ERROR);
        GT_STATIC_ASSERT(JSize > 0, GT_INTERNAL_ERROR);

        T m_values[JSize][ISize];

      public:
        GT_FUNCTION ij_cache_storage() {}

        template <class Accessor>
        GT_FUNCTION T &at(int_t i, int_t j, Accessor const &acc) {
            i += accessor_offset<0>(acc) + IZero;
            j += accessor_offset<1>(acc) + JZero;
            assert(accessor_offset<2>(acc) == 0);
            assert(i >= 0);
            assert(i < ISize);
            assert(j >= 0);
            assert(j < JSize);
            return m_values[j][i];
        }
    };

    template <class Arg, int_t ITile, int_t JTile, class Extent>
    struct make_ij_cache_storage {
        GT_STATIC_ASSERT(ITile > 0, GT_INTERNAL_ERROR);
        GT_STATIC_ASSERT(JTile > 0, GT_INTERNAL_ERROR);
        using type = ij_cache_storage<typename Arg::data_store_t::data_t,
            ITile + Extent::iplus::value - Extent::iminus::value,
            JTile + Extent::jplus::value - Extent::jminus::value,
            -Extent::iminus::value,
            -Extent::jminus::value>;
    };
#else
    template <class T, int_t NumColors, int_t ISize, int_t JSize, int_t IZero, int_t JZero>
    class ij_cache_storage {
        GT_STATIC_ASSERT(ISize > 0, GT_INTERNAL_ERROR);
        GT_STATIC_ASSERT(JSize > 0, GT_INTERNAL_ERROR);
        GT_STATIC_ASSERT(NumColors > 0, GT_INTERNAL_ERROR);

        T m_values[JSize][NumColors][ISize];

      public:
        GT_FUNCTION ij_cache_storage() {}

        template <int_t Color, class Accessor>
        GT_FUNCTION T &at(int_t i, int_t j, Accessor const &acc) {
            i += accessor_offset<0>(acc) + IZero;
            int_t color = Color + accessor_offset<1>(acc);
            j += accessor_offset<2>(acc) + JZero;
            assert(accessor_offset<3>(acc) == 0);
            assert(i >= 0);
            assert(i < ISize);
            assert(color >= 0);
            assert(color < NumColors);
            assert(j >= 0);
            assert(j < JSize);
            return m_values[j][color][i];
        }
    };

    template <class Arg, int_t ITile, int_t JTile, class Extent>
    struct make_ij_cache_storage {
        GT_STATIC_ASSERT(ITile > 0, GT_INTERNAL_ERROR);
        GT_STATIC_ASSERT(JTile > 0, GT_INTERNAL_ERROR);
        using type = ij_cache_storage<typename Arg::data_store_t::data_t,
            Arg::location_t::n_colors::value,
            ITile + Extent::iplus::value - Extent::iminus::value,
            JTile + Extent::jplus::value - Extent::jminus::value,
            -Extent::iminus::value,
            -Extent::jminus::value>;
    };
#endif

    enum class sync_type { fill, flush };

    template <class Arg, class T, int_t Minus, int_t Plus>
    class k_cache_storage {

        GT_STATIC_ASSERT(Minus <= 0, GT_INTERNAL_ERROR);
        GT_STATIC_ASSERT(Plus >= 0, GT_INTERNAL_ERROR);

        T m_values[Plus - Minus + 1];

        template <sync_type SyncType, class Data>
        GT_FUNCTION enable_if_t<SyncType == sync_type::fill> sync_at(Data const &data, int_t k) {
            if (auto *src = data.template deref_for_k_cache<Arg>(k))
                m_values[k - Minus] = *src;
        }

        template <sync_type SyncType, class Data>
        GT_FUNCTION enable_if_t<SyncType == sync_type::flush> sync_at(Data const &data, int_t k) {
            if (auto *dst = data.template deref_for_k_cache<Arg>(k))
                *dst = m_values[k - Minus];
        }

        template <class Policy, sync_type SyncType>
        GT_META_DEFINE_ALIAS(sync_point,
            std::integral_constant,
            (int_t,
                (execute::is_forward<Policy>::value && SyncType == sync_type::fill) ||
                        (execute::is_backward<Policy>::value && SyncType == sync_type::flush)
                    ? Plus
                    : Minus));

      public:
        /**
         * @brief retrieve value in a cache given an accessor for a k cache
         * @param acc the accessor that contains the offsets being accessed
         */
        template <class Accessor>
        GT_FUNCTION T &at(Accessor const &acc) {
            int_t offset = accessor_offset<2>(acc);
            assert(offset >= Minus);
            assert(offset <= Plus);
            assert(accessor_offset<0>(acc) == 0);
            assert(accessor_offset<1>(acc) == 0);
            return m_values[offset - Minus];
        }

        /**
         * @brief slides the values of the ring buffer
         */
        template <class Policy>
        GT_FUNCTION enable_if_t<execute::is_forward<Policy>::value> slide() {
#pragma unroll
            for (int_t k = 0; k < Plus - Minus; ++k)
                m_values[k] = m_values[k + 1];
        }

        template <class Policy>
        GT_FUNCTION enable_if_t<execute::is_backward<Policy>::value> slide() {
#pragma unroll
            for (int_t k = Plus - Minus; k > 0; --k)
                m_values[k] = m_values[k - 1];
        }

        template <class Policy, sync_type SyncType, class Data, int_t SyncPoint = sync_point<Policy, SyncType>::value>
        GT_FUNCTION void sync(Data const &data, bool sync_all) {
            if (sync_all)
#pragma unroll
                for (int_t k = Minus; k <= Plus; ++k)
                    sync_at<SyncType>(data, k);
            else
                sync_at<SyncType>(data, SyncPoint);
        }
    };

    template <class Arg, class Extent>
    struct make_k_cache_storage {
        GT_STATIC_ASSERT(
            Extent::iminus::value == 0, "KCaches can not be use with a non zero extent in the horizontal dimensions");
        GT_STATIC_ASSERT(
            Extent::iplus::value == 0, "KCaches can not be use with a non zero extent in the horizontal dimensions");
        GT_STATIC_ASSERT(
            Extent::jminus::value == 0, "KCaches can not be use with a non zero extent in the horizontal dimensions");
        GT_STATIC_ASSERT(
            Extent::jplus::value == 0, "KCaches can not be use with a non zero extent in the horizontal dimensions");

        GT_STATIC_ASSERT(Extent::kminus::value <= 0, GT_INTERNAL_ERROR);
        GT_STATIC_ASSERT(Extent::kplus::value >= 0, GT_INTERNAL_ERROR);

        using type =
            k_cache_storage<Arg, typename Arg::data_store_t::data_t, Extent::kminus::value, Extent::kplus::value>;
    };
} // namespace gridtools
