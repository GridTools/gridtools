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

#include "../../common/array.hpp"
#include "../../common/generic_metafunctions/accumulate.hpp"
#include "../../common/gt_assert.hpp"
#include "../../meta/type_traits.hpp"
#include "../../meta/utility.hpp"
#include "../block_size.hpp"
#include "../extent.hpp"
#include "../iteration_policy.hpp"
#include "../offset_computation.hpp"
#include "cache_traits.hpp"
#include "meta_storage_cache.hpp"

namespace gridtools {

#ifdef STRUCTURED_GRIDS
    template <class T, int_t ISize, int_t JSize, int_t IZero, int_t JZero>
    class ij_cache_storage {
        GRIDTOOLS_STATIC_ASSERT(ISize > 0, GT_INTERNAL_ERROR);
        GRIDTOOLS_STATIC_ASSERT(JSize > 0, GT_INTERNAL_ERROR);

        T m_values[JSize][ISize];

      public:
        GT_FUNCTION ij_cache_storage() {}

        template <class Accessor>
        GT_FUNCTION T &at(int_t i, int_t j, Accessor const &acc) {
            i += accessor_offset<0>(acc) - IZero;
            j += accessor_offset<1>(acc) - JZero;
            assert(accessor_offset<2>(acc) == 0);
            assert(i >= 0);
            assert(i < ISize);
            assert(j >= 0);
            assert(j < JSize);
            return m_values[j][i];
        }
    };

    template <class T, int_t Size, int_t ZeroOffset>
    class k_cache_storage {
        GRIDTOOLS_STATIC_ASSERT(Size > 0, GT_INTERNAL_ERROR);
        GRIDTOOLS_STATIC_ASSERT(ZeroOffset < Size, GT_INTERNAL_ERROR);

        mutable T m_values[Size];

        struct slide_forward_f {
            T *m_values;
            GT_FUNCTION void operator()() const {
                for (int_t k = 0; k < Size - 1; ++k)
                    m_values[k] = std::move(m_values[k + 1]);
            }
        };

        struct slide_backward_f {
            T *m_values;
            GT_FUNCTION void operator()() const {
                for (int_t k = Size - 1; k > 0; --k)
                    m_values[k] = std::move(m_values[k - 1]);
            }
        };

        template <enumtype::execution, class Src>
        struct fill_f;

        template <class Src>
        struct fill_f<enumtype::forward, Src> {
            Src const &m_src;
            T *m_dst;
            bool m_first_level;
            GT_FUNCTION void operator()() const {
                for (int_t i = m_first_level ? ZeroOffset : Size - 1; i < Size; ++i)
                    m_dst[i] = m_src(i - ZeroOffset);
            }
        };

        template <class Src>
        struct fill_f<enumtype::backward, Src> {
            Src const &m_src;
            T *m_dst;
            bool m_first_level;
            GT_FUNCTION void operator()() const {
                for (int_t i = m_first_level ? ZeroOffset : 0; i >= 0; --i)
                    m_dst[i] = m_src(i - ZeroOffset);
            }
        };

        template <enumtype::execution, class Dst>
        struct flush_f;

        template <class Dst>
        struct flush_f<enumtype::forward, Dst> {
            T *m_src;
            Dst &m_dst;
            bool m_last_level;
            GT_FUNCTION void operator()() const {
                for (int_t i = m_last_level ? ZeroOffset : 0; i >= 0; --i)
                    m_dst(i - ZeroOffset) = std::move(m_src[i]);
            }
        };

        template <class Dst>
        struct flush_f<enumtype::backward, Dst> {
            T *m_src;
            Dst &m_dst;
            bool m_last_level;
            GT_FUNCTION void operator()() const {
                for (int_t i = m_last_level ? ZeroOffset : Size - 1; i < Size; ++i)
                    m_src[i] = m_dst(i - ZeroOffset);
            }
        };

      public:
        /**
         * @brief retrieve value in a cache given an accessor for a k cache
         * @param acc the accessor that contains the offsets being accessed
         */
        template <class Accessor>
        GT_FUNCTION T &at(Accessor const &acc) const {
            int_t offset = accessor_offset<2>(acc) + ZeroOffset;
            assert(offset >= 0);
            assert(offset < Size);
            assert(accessor_offset<0>(acc) == 0);
            assert(accessor_offset<1>(acc) == 0);
            return m_values[offset];
        }

        /**
         * @brief slides the values of the ring buffer
         */
        template <enumtype::execution Policy>
        GT_FUNCTION constexpr conditional_t<Policy == enumtype::forward, slide_forward_f, slide_backward_f>
        slide() const {
            return {m_values};
        }

        template <enumtype::execution Policy, class Src>
        GT_FUNCTION constexpr fill_f<Policy, Src> fill(bool first_level, Src const &src) const {
            return {src, m_values, first_level};
        }

        template <enumtype::execution Policy, class Dst>
        GT_FUNCTION constexpr flush_f<Policy, Dst> flush(bool last_level, Dst const &dst) const {
            return {m_values, dst, last_level};
        }

        template <enumtype::execution Policy, class Dst>
        GT_FUNCTION enable_if_t<Policy == enumtype::backward> flush(bool last_level, Dst const &dst) const {
            for (int_t i = last_level ? ZeroOffset : Size - 1; i < Size; ++i)
                m_values[i] = dst(i - ZeroOffset);
        }
    };

#else
    template <class T, int_t NumColors, int_t ISize, int_t JSize, int_t IZero, int_t JZero>
    class ij_cache_storage {
        GRIDTOOLS_STATIC_ASSERT(ISize > 0, GT_INTERNAL_ERROR);
        GRIDTOOLS_STATIC_ASSERT(JSize > 0, GT_INTERNAL_ERROR);
        GRIDTOOLS_STATIC_ASSERT(NumColors > 0, GT_INTERNAL_ERROR);

        T m_values[JSize][NumColors][ISize];

      public:
        GT_FUNCTION ij_cache_storage() {}

        template <int_t Color, class Accessor>
        GT_FUNCTION T &at(int_t i, int_t j, Accessor const &acc) {
            GRIDTOOLS_STATIC_ASSERT(Color > 0, GT_INTERNAL_ERROR);
            GRIDTOOLS_STATIC_ASSERT(Color < NumColors, GT_INTERNAL_ERROR);
            i += accessor_offset<0>(acc) - IZero;
            int_t color = Color + accessor_offset<1>(acc);
            j += accessor_offset<2>(acc) - JZero;
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
#endif

    template <class Cache, class BlockSize, class Extent, class Arg>
    struct cache_storage;

    template <cache_io_policy CacheIoPolicy,
        uint_t TileI,
        uint_t TileJ,
        int_t IMinus,
        int_t IPlus,
        int_t JMinus,
        int JPlus,
        int_t KMinus,
        int_t KPlus,
        class Arg>
    struct cache_storage<detail::cache_impl<IJ, Arg, CacheIoPolicy>,
        block_size<TileI, TileJ, 1>,
        extent<IMinus, IPlus, JMinus, JPlus, KMinus, KPlus>,
        Arg> {
        GRIDTOOLS_STATIC_ASSERT(TileI > 0 && TileJ > 0, GT_INTERNAL_ERROR);
        GRIDTOOLS_STATIC_ASSERT(KMinus == 0 && KPlus == 0, "Only KCaches can be accessed with a non null extent in K");

        using value_type = typename Arg::data_store_t::data_t;

// TODO ICO_STORAGE in irregular grids we have one more dim for color
#ifndef STRUCTURED_GRIDS
        static constexpr int extra_dims = 1;

        using meta_t = meta_storage_cache<layout_map<3, 2, 1, 0>,
            IPlus - IMinus + TileI,
            Arg::location_t::n_colors::value,
            JPlus - JMinus + TileJ,
            1>;

        static constexpr uint_t total_length =
            (IPlus - IMinus + TileI) * (JPlus - JMinus + TileJ) * Arg::location_t::n_colors::value;

#else
        static constexpr int extra_dims = 0;
        using meta_t = meta_storage_cache<layout_map<2, 1, 0>, IPlus - IMinus + TileI, JPlus - JMinus + TileJ, 1>;
        static constexpr uint_t total_length = (IPlus - IMinus + TileI) * (JPlus - JMinus + TileJ);
#endif

        GT_FUNCTION cache_storage() {}

        template <uint_t Color, typename Accessor>
        GT_FUNCTION value_type &RESTRICT at(array<int, 2> const &thread_pos, Accessor const &acc) {
            int_t offset = (thread_pos[0] - IMinus + accessor_offset<0>(acc)) * meta_t::template stride<0>() +
                           (thread_pos[1] - JMinus + accessor_offset<1 + extra_dims>(acc)) *
                               meta_t::template stride<1 + extra_dims>() +
                           extra_dims * (Color + accessor_offset<1>(acc)) * meta_t::template stride<1>();
            assert(accessor_offset<2 + extra_dims>(acc) == 0);
            assert(offset < total_length);
            return m_values[offset];
        }

      private:
        value_type m_values[total_length];
    };

    template <cache_io_policy CacheIoPolicy,
        int_t IMinus,
        int_t IPlus,
        int_t JMinus,
        int JPlus,
        int_t KMinus,
        int_t KPlus,
        class Arg>
    struct cache_storage<detail::cache_impl<K, Arg, CacheIoPolicy>,
        block_size<1, 1, 1>,
        extent<IMinus, IPlus, JMinus, JPlus, KMinus, KPlus>,
        Arg> {
        using cache_t = detail::cache_impl<K, Arg, CacheIoPolicy>;

        static constexpr int_t kminus = KMinus;
        static constexpr int_t kplus = KPlus;

        using value_type = typename Arg::data_store_t::data_t;

        GRIDTOOLS_STATIC_ASSERT(IMinus == 0 && JMinus == 0 && IPlus == 0 && JPlus == 0,
            "KCaches can not be use with a non null extent in the horizontal dimensions");

        static constexpr int_t total_length = KPlus - KMinus + 1;
        GRIDTOOLS_STATIC_ASSERT(total_length > 0, GT_INTERNAL_ERROR);

        /**
         * @brief retrieve value in a cache given an accessor for a k cache
         * @param acc the accessor that contains the offsets being accessed
         */
        template <class Accessor>
        GT_FUNCTION value_type &at(Accessor const &acc) const {
            int_t offset = accessor_offset<2>(acc) - KMinus;
            assert(offset >= 0);
            assert(offset < total_length);
            assert(accessor_offset<0>(acc) == 0);
            assert(accessor_offset<1>(acc) == 0);
            return m_values[offset];
        }

        /**
         * @brief slides the values of the ring buffer
         */
        template <enumtype::execution Policy>
        GT_FUNCTION enable_if_t<Policy == enumtype::forward> slide() {
            constexpr auto end = total_length - 1;
            for (int_t k = 0; k < end; ++k)
                m_values[k] = m_values[k + 1];
        }

        template <enumtype::execution Policy>
        GT_FUNCTION enable_if_t<Policy == enumtype::backward> slide() {
            for (int_t k = total_length - 1; k > 0; --k)
                m_values[k] = m_values[k - 1];
        }

      private:
        mutable value_type m_values[total_length];
    };
} // namespace gridtools
