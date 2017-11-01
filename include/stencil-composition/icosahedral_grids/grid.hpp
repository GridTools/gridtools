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
#include "stencil-composition/grid_base.hpp"
#include "stencil-composition/icosahedral_grids/icosahedral_topology.hpp"
#include "../../common/gpu_clone.hpp"
#include "unstructured_mesh.hpp"
#include "topology_traits.hpp"

namespace gridtools {

    /* functor class that will apply the clone to the gpu of a mesh.
     * Contains two specializations depending on the type of topology.
     * If the topology is not of unstructured mesh type, the clone operation is ignored
     */
    struct mesh_cloner {
      public:
        template < typename T, typename boost::enable_if< typename is_unstructured_mesh< T >::type, int >::type = 0 >
        T *apply(T *umesh) const {
            umesh->clone_to_device();
            return umesh->gpu_object_ptr;
        }

        template < typename T, typename boost::disable_if< typename is_unstructured_mesh< T >::type, int >::type = 0 >
        T *apply(T *umesh) const {
            return umesh;
        }
    };

    struct empty {};

    template < typename Axis, typename GridTopology >
    struct grid : public grid_base< Axis >, public clonable_to_gpu< grid< Axis, GridTopology > > {
        GRIDTOOLS_STATIC_ASSERT((is_interval< Axis >::value), GT_INTERNAL_ERROR);
        GRIDTOOLS_STATIC_ASSERT((is_grid_topology< GridTopology >::value), GT_INTERNAL_ERROR);

        typedef GridTopology grid_topology_t;

        using unstructured_mesh_t =
            typename boost::mpl::if_c< is_unstructured_mesh< GridTopology >::value, GridTopology, empty >::type;

      private:
        unstructured_mesh_t *m_umesh;

      public:
        static constexpr enumtype::grid_type c_grid_type = enumtype::icosahedral;

        // ctr API of the grid that uses an unstructured mesh topology, where the require
        // mesh object is passed by argument
        template < typename T = GridTopology,
            typename boost::enable_if_c< boost::is_same< T, GridTopology >::value &&
                                             is_unstructured_mesh< GridTopology >::value,
                int >::type = 0 >
        explicit grid(const array< uint_t, 5 > &i, const array< uint_t, 5 > &j, GridTopology &umesh)
            : grid_base< Axis >(halo_descriptor(i[minus], i[plus], i[begin], i[end], i[length]),
                  halo_descriptor(j[minus], j[plus], j[begin], j[end], j[length])),
              m_umesh(&umesh) {}

        // ctr API of the grid that uses an structured topology. Since no runtime
        // mesh information is required, the ctr only accepts domain size properties
        template < typename T = GridTopology,
            typename boost::enable_if_c< boost::is_same< T, GridTopology >::value &&
                                             !is_unstructured_mesh< GridTopology >::value,
                int >::type = 0 >
        explicit grid(const array< uint_t, 5 > &i, const array< uint_t, 5 > &j)
            : grid_base< Axis >(halo_descriptor(i[minus], i[plus], i[begin], i[end], i[length]),
                  halo_descriptor(j[minus], j[plus], j[begin], j[end], j[length])) {}

        void clone_to_device() {
            m_umesh = mesh_cloner().apply(m_umesh);
            clonable_to_gpu< grid< Axis, GridTopology > >::clone_to_device();
        }
        GT_FUNCTION_DEVICE grid(grid const &other) : grid_base< Axis >(other), m_umesh(other.m_umesh) {}

        GT_FUNCTION
        unstructured_mesh_t &umesh() {
            assert(m_umesh);
            return *m_umesh;
        }

        GT_FUNCTION
        unstructured_mesh_t const &grid_topology() const {
            assert(m_umesh);
            return *m_umesh;
        }
    };

    template < typename Grid >
    struct is_grid : boost::mpl::false_ {};

    template < typename Axis, typename GridTopology >
    struct is_grid< grid< Axis, GridTopology > > : boost::mpl::true_ {};
}
