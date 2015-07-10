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

#include <gridtools.hpp>
#include <stencil-composition/backend.hpp>

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


    // template <uint_t N>
    // struct factorial{
    //     uint_t value=N*factorial<N-1>::value;
    // };

    // template<>
    // struct factorial<0>{
    //     uint_t value=1;
    // };

    // template <uint_t n, uint_t m>
    // struct combinations{
    //     static_assert(n>m, "wrong combination");
    //     uint_t value = (uint_t) factorial<n>::value/(factorial<m>::value * factorial<n-m>::value);
    // };

    template <typename LayoutType>
    using storage_t = gridtools::intrepid_storage<typename gridtools::BACKEND::storage_type<float_type, LayoutType >::type>;
    using storage3=storage_t<layout_map<0,1,2> >;
    using storage4=storage_t<layout_map<0,1,2,3> >;
    // using storage2=storage_t<layout_map<0,1> >;
    // using storage1=storage_t<layout_map<0> >;

    // create cubature factory
    static Intrepid::DefaultCubatureFactory<double, storage3> cubFactory;

    // template<uint_t dimension, uint_t order>
    // struct tensor_product_element{

    //     template<uint_t codimension>
    //     using  bondary_w_codim = tensor_product_element<dimension-codimension, order>;

    //     template<uint_t codimension>
    //     using n_boundary_w_codim= static_uint<(pow<dimension-codimension>(2))*combinations<dimension, codimension >::value>;

    //     using n_vertices = static_uint<pow<dimension>(2)>;
    //     using n_points = static_uint<pow<dimension>(2+(order-1))>;

    //     template<uint_t codimension>
    //     using n_internal_points = static_uint<n_points::value-(boundary_w_codim<codimension>::n_points::value*n_boundaries_w_codim<codimension>::value)>;

    //     template<uint_t codimension>
    //     using n_boundary_points=static_uint<n_points::value - n_internal_points<codimension>::value >;

    // };

    // static_assert(tensor_product_element<3,1>::n_vertices::value==4);
    // static_assert(tensor_product_element<3,1>::n_internal_points<1>::value==0);
    // static_assert(tensor_product_element<3,1>::n_boundary_points<1>::value==4);
    // static_assert(tensor_product_element<3,2>::n_points::value==4);

    // static_assert(tensor_product_element<3,2>::n_vertices::value==4);
    // static_assert(tensor_product_element<3,2>::n_internal_points<1>::value==1);
    // static_assert(tensor_product_element<3,2>::n_boundary_points<1>::value==26);
    // static_assert(tensor_product_element<3,2>::n_points::value==27);
    // static_assert(tensor_product_element<3,2>::n_boundary_points<2>::value==20);
    // static_assert(tensor_product_element<3,2>::n_boundary_points<3>::value==4);

    namespace fe{
        using namespace Intrepid;
        static const shards::CellTopology cellType = shards::getCellTopologyData< shards::Hexahedron<> >(); // cell type: hexahedron

        // using hypercube_t=tensor_product_element<3,2>;
        static const Basis_HGRAD_HEX_C1_FEM<double, storage3 > hexBasis;                       // create hex basis
        static const /*constexpr*/ int spaceDim = cellType.getDimension();                                                // retrieve spatial dimension
        static const /*constexpr*/ int numNodes = cellType.getNodeCount();                                                // retrieve number of 0-cells (nodes)
        static const /*constexpr*/ int basisCardinality = hexBasis.getCardinality();                                              // get basis cardinality
        // set cubature degree, e.g. 2
        static const int cubDegree = basisCardinality;

        // create default cubature
        static const Teuchos::RCP<Cubature<double, storage3> > myCub = cubFactory.create(fe::cellType, cubDegree);
        // retrieve number of cubature points
        static const int numCubPoints = myCub->getNumPoints();

    } //namespace fe

    //geometric map: in the isoparametric case theis namespace is the same as the previous one
    namespace geo_map{
        using namespace Intrepid;
        static const shards::CellTopology cellType = shards::getCellTopologyData< shards::Hexahedron<> >(); // cell type: hexahedron

        // using hypercube_t=tensor_product_element<3,2>;
        static const Basis_HGRAD_HEX_C1_FEM<double, storage3 > hexBasis;                       // create hex basis
        static const /*constexpr*/ int spaceDim = cellType.getDimension();
        // retrieve spatial dimension
        static const /*constexpr*/ int numNodes = cellType.getNodeCount();
        // retrieve number of 0-cells (nodes)
        static const /*constexpr*/ int basisCardinality = hexBasis.getCardinality();
        // get basis cardinality

        // set cubature degree, e.g. 2
        static const int cubDegree = basisCardinality;

        // create default cubature
        static const Teuchos::RCP<Cubature<double, storage3> > myCub = cubFactory.create(fe::cellType, cubDegree);
        // retrieve number of cubature points
        static const int numCubPoints = myCub->getNumPoints();

    } //namespace fe
}//namespace gridtools
