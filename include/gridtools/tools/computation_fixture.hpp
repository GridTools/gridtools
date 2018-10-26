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

#include <utility>

#include <gtest/gtest.h>

#include "../common/defs.hpp"
#include "../common/halo_descriptor.hpp"
#include "../common/selector.hpp"
#include "../stencil-composition/axis.hpp"
#include "../stencil-composition/grid.hpp"
#include "../stencil-composition/make_computation.hpp"
#include "../storage/common/halo.hpp"
#include "../storage/storage-facility.hpp"
#include "backend_select.hpp"
#include "verifier.hpp"

namespace gridtools {
    template <size_t HaloSize = 0, class Axis = axis<1>>
    class computation_fixture : virtual public ::testing::Test {
        uint_t m_d1;
        uint_t m_d2;
        uint_t m_d3;

      public:
        static constexpr uint_t halo_size = HaloSize;
        using storage_tr = storage_traits<backend_t::backend_id_t>;
        using halo_t = halo<HaloSize, HaloSize, 0>;
        using storage_info_t = storage_tr::storage_info_t<0, 3, halo_t>;
        using j_storage_info_t = storage_tr::special_storage_info_t<1, selector<0, 1, 0>>;
        using scalar_storage_info_t = storage_tr::special_storage_info_t<2, selector<0, 0, 0>>;

        using storage_type = storage_tr::data_store_t<float_type, storage_info_t>;
        using j_storage_type = storage_tr::data_store_t<float_type, j_storage_info_t>;
        using scalar_storage_type = storage_tr::data_store_t<float_type, scalar_storage_info_t>;

        computation_fixture(uint_t d1, uint_t d2, uint_t d3) : m_d1(d1), m_d2(d2), m_d3(d3) {}

        uint_t d1() const { return m_d1; }
        uint_t d2() const { return m_d2; }
        uint_t d3() const { return m_d3; }

        uint_t &d1() { return m_d1; }
        uint_t &d2() { return m_d2; }
        uint_t &d3() { return m_d3; }

        halo_descriptor i_halo_descriptor() const {
            return {halo_size, halo_size, halo_size, m_d1 - halo_size - 1, m_d1};
        }
        halo_descriptor j_halo_descriptor() const {
            return {halo_size, halo_size, halo_size, m_d2 - halo_size - 1, m_d2};
        }

        auto make_grid() const
            GT_AUTO_RETURN(::gridtools::make_grid(i_halo_descriptor(), j_halo_descriptor(), Axis{m_d3}));

        template <class Storage = storage_type, class T>
        Storage make_storage(T &&obj) const {
            return {{m_d1, m_d2, m_d3}, std::forward<T>(obj)};
        }

        template <class... Args>
        auto make_computation(Args &&... args)
            GT_AUTO_RETURN(::gridtools::make_computation<backend_t>(make_grid(), std::forward<Args>(args)...));

        template <class Expected, class Actual>
        void verify(Expected const &expected, Actual const &actual) const {
            EXPECT_TRUE(verifier{}.verify(
                make_grid(), expected, actual, {{{halo_size, halo_size}, {halo_size, halo_size}, {0, 0}}}));
        }
    };
} // namespace gridtools
