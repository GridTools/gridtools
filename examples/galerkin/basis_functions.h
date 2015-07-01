#pragma once

#include <Intrepid_Basis.hpp>
#include <Intrepid_Types.hpp>

#include "Intrepid_HGRAD_TET_Cn_FEM_ORTH.hpp"
#include "Intrepid_HGRAD_TET_Cn_FEM.hpp"
#include "Intrepid_HGRAD_TRI_Cn_FEM.hpp"
#include "Intrepid_HGRAD_LINE_Cn_FEM.hpp"
#include "Intrepid_HGRAD_TET_C1_FEM.hpp"
#include "Intrepid_HGRAD_TRI_C1_FEM.hpp"
#include "Intrepid_HGRAD_LINE_C1_FEM.hpp"
#include "Intrepid_HGRAD_TET_C2_FEM.hpp"
#include "Intrepid_HGRAD_TRI_C2_FEM.hpp"
#include "Intrepid_HGRAD_HEX_C2_FEM.hpp"
#include "Intrepid_HGRAD_QUAD_C2_FEM.hpp"
#include "Intrepid_HGRAD_HEX_Cn_FEM.hpp"
#include "Intrepid_HGRAD_QUAD_Cn_FEM.hpp"
#include "intrepid_storage.h"

#include "Shards_CellTopology.hpp"
#include "Shards_CellTopologyData.h"
#include "Shards_BasicTopologies.hpp"

#include <gridtools.h>
#include <stencil-composition/backend.h>

namespace gridtools{
#ifdef CUDA_EXAMPLE
#define BACKEND backend<enumtype::Cuda, enumtype::Block >
#else
#ifdef BACKEND_BLOCK
#define BACKEND backend<enumtype::Host, enumtype::Block >
#else
#define BACKEND backend<enumtype::Host, enumtype::Naive >
#endif
#endif

    template <typename LayoutType>
    using storage_t = gridtools::intrepid_storage<typename gridtools::BACKEND::storage_type<float_type, LayoutType >::type>;
    using storage3=storage_t<layout_map<0,1,2> >;

    namespace fe{
        using namespace Intrepid;
        static const shards::CellTopology cellType = shards::getCellTopologyData< shards::Hexahedron<> >(); // cell type: hexahedron

        static const Basis_HGRAD_HEX_C1_FEM<double, storage3 > hexBasis;                       // create hex basis
        static const int spaceDim = cellType.getDimension();                                                // retrieve spatial dimension
        static const int numNodes = cellType.getNodeCount();                                                // retrieve number of 0-cells (nodes)

        static const int numFields = hexBasis.getCardinality();                                              // get basis cardinality
    } //namespace fe
}//namespace gridtools
