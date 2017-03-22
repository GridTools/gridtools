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
#pragma once
#include "common/defs.hpp"
#include "common/halo_descriptor.hpp"
#include "cell_topology.hpp"
#include "gt_math.hpp"

/**
@file
@brief Simple Partitioner Class
This file defines a simple partitioner splitting a structured cartesian grid
*/
namespace gridtools {

    struct partitioner_dummy;
    template < typename GridTopology, typename Communicator >
    class partitioner_trivial;

    template < typename Derived >
    struct space_dimensions;

    template <>
    struct space_dimensions< partitioner_dummy > {
        static const ushort_t value = 3;
    };

    template < typename TopologyType, typename Communicator >
    struct space_dimensions< partitioner_trivial< TopologyType, Communicator > > {
        static const ushort_t value = TopologyType::space_dimensions;
    };

    template < typename Derived >
    class partitioner {

      public:
        enum Flag {
            UP = 1,
            LOW =
#ifdef CXX11_ENABLED
                gt_pow< space_dimensions< Derived >::value >::apply(2)
#else
                8 // 2^3, 3D topology
#endif
        };

        /**@brief constructor
           suppose we are using an MPI cartesian communicator:
           then we have a coordinates (e.g. the local i,j,k identifying a processor id) and dimensions (e.g. IxJxK)
        */
        GT_FUNCTION
        partitioner() {}
    };

    /**@brief dummy partitioner, must be empty (only static data)

       used in the case in which no partitioner is needed
     */
    class partitioner_dummy : public partitioner< partitioner_dummy > {

      public:
        typedef partitioner< partitioner_dummy > super;

        static int boundary() { return 64 + 32 + 16 + 8 + 4 + 2 + 1; }
        GT_FUNCTION
        static bool at_boundary(ushort_t const & /*coordinate*/, super::Flag const & /*flag_*/) { return true; }
        static const ushort_t space_dimensions = 3;
    };

    template < typename Partitioner >
    struct is_partitioner_dummy : boost::false_type {};

    template <>
    struct is_partitioner_dummy< partitioner_dummy > : boost::true_type {};

} // namespace gridtools
