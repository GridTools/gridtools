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

#include "../../common/array.hpp"
#include "../../common/gt_assert.hpp"
#include "../location_type.hpp"
#include "../../common/array_addons.hpp"
#include "../../common/gpu_clone.hpp"
#include "../../common/generic_metafunctions/pack_get_elem.hpp"
#include "../../common/generic_metafunctions/gt_integer_sequence.hpp"

#include "icosahedral_topology_metafunctions.hpp"

#include "atlas/mesh/Mesh.h"
#include "atlas/mesh/HybridElements.h"
#include "atlas/mesh/Nodes.h"

namespace gridtools {

#ifdef ENABLE_ATLAS
    class unstructured_mesh : public clonable_to_gpu< unstructured_mesh > {
      public:
        enum class mb_connectivity_type {
            cell_to_cell = 0,
            cell_to_edge,
            cell_to_vertex,
            edge_to_cell,
            edge_to_edge,
            edge_to_vertex,
        };

        enum class irr_connectivity_type { vertex_to_cell = 0, vertex_to_edge, vertex_to_vertex };

        using cells = enumtype::cells;
        using edges = enumtype::edges;
        using vertices = enumtype::vertices;

      private:
        size_t m_ncells;
        size_t m_nedges;
        size_t m_nvertices;

        array< atlas::mesh::MultiBlockConnectivityImpl *, 6 > m_mb_connectivity;
        array< atlas::mesh::IrregularConnectivityImpl *, 3 > m_irr_connectivity;

      public:
        unstructured_mesh() = delete;

        unstructured_mesh(const unstructured_mesh &o) = default;

        GT_FUNCTION
        atlas::mesh::MultiBlockConnectivityImpl &connectivity(mb_connectivity_type ctype) {
            assert(m_mb_connectivity[(uint_t)ctype]);
            return *(m_mb_connectivity[(uint_t)ctype]);
        }

        GT_FUNCTION
        atlas::mesh::IrregularConnectivityImpl &connectivity(irr_connectivity_type ctype) {
            assert(m_irr_connectivity[(uint_t)ctype]);
            return *(m_irr_connectivity[(uint_t)ctype]);
        }

        void clone_to_device() {
            for (size_t i = 0; i != m_mb_connectivity.size(); ++i) {
                m_mb_connectivity[i]->cloneToDevice();
                m_mb_connectivity[i] = m_mb_connectivity[i]->gpu_object_ptr();
            }
            for (size_t i = 0; i != m_irr_connectivity.size(); ++i) {
                m_irr_connectivity[i]->cloneToDevice();
                m_irr_connectivity[i] = m_irr_connectivity[i]->gpu_object_ptr();
            }
            clonable_to_gpu< unstructured_mesh >::clone_to_device();
        }

        unstructured_mesh(atlas::Mesh &mesh)
            : m_ncells(mesh.cells().size()), m_nedges(mesh.edges().size()), m_nvertices(mesh.nodes().size()),
              m_mb_connectivity{&(mesh.cells().cell_connectivity()),
                  &(mesh.cells().edge_connectivity()),
                  &(mesh.cells().node_connectivity()),
                  &(mesh.edges().cell_connectivity()),
                  &(mesh.edges().edge_connectivity()),
                  &(mesh.edges().node_connectivity())},
              m_irr_connectivity{&(mesh.nodes().cell_connectivity()),
                  &(mesh.nodes().edge_connectivity()),
                  (mesh.nodes().has_connectivity("nodes") ? &(mesh.nodes().connectivity("nodes")) : NULL)} {}
    };
#endif

    class trivial_umesh {};

} // namespace gridtools
