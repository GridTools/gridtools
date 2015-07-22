#pragma once

//! [includes]
#include <Intrepid_Basis.hpp>
#include <Intrepid_Types.hpp>
#include <Intrepid_FieldContainer.hpp>

#include <Intrepid_Cubature.hpp>
#include <Intrepid_DefaultCubatureFactory.hpp>

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

#include "Shards_CellTopology.hpp"
#include "Shards_CellTopologyData.h"
#include "Shards_BasicTopologies.hpp"

#include <gridtools.hpp>
#include <stencil-composition/backend.hpp>
//! [includes]

//! [wrapper]
#include "intrepid_storage.h"
#include "tensor_product_element.h"
//! [wrapper]

namespace gridtools{
//! [storage definition]
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
    using storage_t = gridtools::intrepid_storage<
        typename gridtools::BACKEND::storage_type<float_type, LayoutType >::type >
        ;
//! [storage definition]

//! [fe namespace]
    namespace fe{
        using namespace Intrepid;
        typedef layout_map<2,1,0> layout_t;

        static const shards::CellTopology cellType = shards::getCellTopologyData< shards::Hexahedron<> >(); // cell type: hexahedron

        static const Basis_HGRAD_HEX_C1_FEM<double, storage_t<layout_map<0,1,2> >// Intrepid::FieldContainer<double>
                                            > hexBasis;                       // create hex basis
        static const uint_t order=1;
        static const /*constexpr*/ int spaceDim = cellType.getDimension();                                                // retrieve spatial dimension
        static const /*constexpr*/ int numNodes = cellType.getNodeCount();                                                // retrieve number of 0-cells (nodes)
        static const /*constexpr*/ int basisCardinality = hexBasis.getCardinality();                                              // get basis cardinality

        //! [tensor product]
        //assert(spaceDim==3);
        using hypercube_t = tensor_product_element<3,1>;
        //! [tensor product]
    } //namespace fe
//! [fe namespace]

    //geometric map: in the isoparametric case theis namespace is the same as the previous one
//! [geomap namespace]
    namespace geo_map{
        using namespace Intrepid;
        static const shards::CellTopology cellType = shards::getCellTopologyData< shards::Hexahedron<> >(); // cell type: hexahedron

        // using hypercube_t=tensor_product_element<3,2>;
        static const Basis_HGRAD_HEX_C1_FEM<double, storage_t<layout_map<0,1,2> >// Intrepid::FieldContainer<double>
                                            > hexBasis;                       // create hex basis
        static const uint_t order=1;
        static const /*constexpr*/ int spaceDim = cellType.getDimension();
        // retrieve spatial dimension
        static const /*constexpr*/ int numNodes = cellType.getNodeCount();
        // retrieve number of 0-cells (nodes)
        static const /*constexpr*/ int basisCardinality = hexBasis.getCardinality();
        // get basis cardinality

    } //namespace geo_map
//! [geomap namespace]


//! [quadrature]
    namespace cubature{
        // set cubature degree, e.g. 2
        static const int cubDegree = fe::order+1;
        // create cubature factory
        static Intrepid::DefaultCubatureFactory<double, storage_t<layout_map<0,1,2> >// Intrepid::FieldContainer<double>
                                                > cubFactory;
    // create default cubature
        static const Teuchos::RCP<Intrepid::Cubature<double, storage_t<layout_map<0,1,2> >// Intrepid::FieldContainer<double>
                                                     > > cub = cubFactory.create(fe::cellType, cubDegree);
        // retrieve number of cubature points
        static const int numCubPoints = cub->getNumPoints();
    }
//! [quadrature]

}//namespace gridtools
