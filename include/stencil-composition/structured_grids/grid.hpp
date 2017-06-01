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
#include "../grid_base.hpp"

namespace gridtools {

    template < typename Axis, typename Partitioner = partitioner_dummy >
    struct grid : public grid_base< Axis, Partitioner >, public clonable_to_gpu< grid< Axis, Partitioner > > {
        static constexpr enumtype::grid_type c_grid_type = enumtype::structured;

        GT_FUNCTION
        explicit grid(halo_descriptor const &direction_i, halo_descriptor const &direction_j)
            : grid_base< Axis, Partitioner >(direction_i, direction_j) {}

        template < typename ParallelStorage >
        GT_FUNCTION explicit grid(const Partitioner &part_, ParallelStorage const &storage_)
            : grid_base< Axis, Partitioner >(part_, storage_) {}

        GT_FUNCTION grid(const grid< Axis, Partitioner > &other) : grid_base< Axis, Partitioner >(other) {}

        // TODO should be removed (use ctor with halo_descriptor)
        GT_FUNCTION
        explicit grid(uint_t *i, uint_t *j) : grid_base< Axis, Partitioner >(i, j) {}
    };

    template < typename Grid >
    struct is_grid : boost::mpl::false_ {};

    template < typename Axis >
    struct is_grid< grid< Axis > > : boost::mpl::true_ {};

    template < typename Axis, typename Partitioner >
    struct is_grid< grid< Axis, Partitioner > > : boost::mpl::true_ {};
}
