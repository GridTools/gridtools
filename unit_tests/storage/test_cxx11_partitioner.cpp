/*
  GridTools Libraries

  Copyright (c) 2016, GridTools Consortium
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
#include "gtest/gtest.h"
#include <stencil-composition/make_computation.hpp>
#include <storage/partitioner_trivial.hpp>

using namespace gridtools;

struct comm {
    static const uint_t ndims = 5;
    gridtools::array< bool, ndims > const &periodic() const { return m_periodic; }
    gridtools::array< int, ndims > const &dimensions() const { return m_dimensions; }
    gridtools::array< int, ndims > const &coordinates() const { return m_coordinates; }
    gridtools::array< bool, ndims > m_periodic;
    gridtools::array< int, ndims > m_dimensions;
    gridtools::array< int, ndims > m_coordinates;
};

TEST(partitioner_trivial, test_partitioner) {
    using namespace enumtype;

    // 5D processor grid 3x3x3x1x5, periodic in the 3rd and 5th dimensions. I am the
    // process in position (0,1,2,0,4)
    comm comm_{{false, false, true, false, true}, {3, 3, 3, 1, 5}, {0, 1, 2, 0, 4}};

    array< ushort_t, 5 > padding{6, 7, 8, 9, 10};
    array< ushort_t, 5 > halo{1, 2, 3, 4, 5};

    // 5D cartesian topology
    typedef partitioner_trivial< cell_topology< topology::cartesian< layout_map< 0, 1, 2, 3, 4 > > >, comm > party;
    party part_(comm_, halo, padding);
    bool success = true;
    success = success && part_.boundary() == 316;

    success = success && part_.at_boundary(0, party::LOW) == true;
    success = success && part_.at_boundary(0, party::UP) == false;
    success = success && part_.at_boundary(1, party::LOW) == false;
    success = success && part_.at_boundary(1, party::UP) == false;
    success = success && part_.at_boundary(2, party::LOW) == false;
    success = success && part_.at_boundary(2, party::UP) == true;
    success = success && part_.at_boundary(3, party::LOW) == true; // only 1 dimension in this component
    success = success && part_.at_boundary(3, party::UP) == true;
    success = success && part_.at_boundary(4, party::LOW) == false;
    success = success && part_.at_boundary(4, party::UP) == true;

    success = success && part_.compute_halo(0, party::LOW) == 6; // global boundary, returns padding
    success = success && part_.compute_halo(0, party::UP) == 1;
    success = success && part_.compute_halo(1, party::LOW) == 2;
    success = success && part_.compute_halo(1, party::UP) == 2;
    success = success && part_.compute_halo(2, party::LOW) == 3;
    success = success && part_.compute_halo(2, party::UP) == 3;  // not returning the padding because periodic
    success = success && part_.compute_halo(3, party::LOW) == 9; // padding, because only 1 dim. and not periodic
    success = success && part_.compute_halo(3, party::UP) == 9;  // padding, because only 1 dim. and not periodic
    success = success && part_.compute_halo(4, party::LOW) == 5;
    success = success && part_.compute_halo(4, party::UP) == 5; // not returning the padding because periodic

    ASSERT_TRUE(success);
}
