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

#include <cassert>
#include <type_traits>
#include <utility>
#include <vector>

#include <gridtools/common/array.hpp>
#include <gridtools/common/defs.hpp>
#include <gridtools/stencil-composition/icosahedral_grids/icosahedral_topology.hpp>
#include <gridtools/stencil-composition/location_type.hpp>

namespace gridtools {
    namespace _impl {
        struct neighbour {
            int_t i, c, j, k;
            template <class Fun>
            auto call(Fun &&fun) const GT_AUTO_RETURN(std::forward<Fun>(fun)(i, c, j, k));
        };

        template <class FromLocation, class ToLocation>
        std::vector<array<int_t, 4>> get_offsets(
            uint_t, std::integral_constant<uint_t, FromLocation::n_colors::value>) {
            assert(false);
            return {};
        }

        template <class FromLocation,
            class ToLocation,
            uint_t C = 0,
            enable_if_t<(C < FromLocation::n_colors::value), int> = 0>
        std::vector<array<int_t, 4>> get_offsets(uint_t c, std::integral_constant<uint_t, C> = {}) {
            if (c > C) {
                return get_offsets<FromLocation, ToLocation>(c, std::integral_constant<uint_t, C + 1>{});
            }
            auto offsets = connectivity<FromLocation, ToLocation, C>::offsets();
            return std::vector<array<int_t, 4>>(offsets.begin(), offsets.end());
        }
    } // namespace _impl

    template <class FromLocation, class ToLocation>
    std::vector<_impl::neighbour> neighbours_of(int_t i, int_t c, int_t j, int_t k) {
        assert(c >= 0);
        assert(c < FromLocation::n_colors::value);
        std::vector<_impl::neighbour> res;
        for (auto &item : _impl::get_offsets<FromLocation, ToLocation>(c))
            res.push_back({i + item[0], c + item[1], j + item[2], k + item[3]});
        return res;
    };
} // namespace gridtools
