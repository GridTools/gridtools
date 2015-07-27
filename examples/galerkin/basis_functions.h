/** \file */
#pragma once

//! [includes]

#include <gridtools.hpp>
#include <stencil-composition/backend.hpp>
//! [includes]

#include "tensor_product_element.h"

#include "element_traits.h"

#include "cell.h"
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
    using storage_t = typename gridtools::BACKEND::storage_type<float_type, LayoutType >::type;
//! [storage definition]
//! [fe namespace]
    template <ushort_t Order, enumtype::Basis BasisType, enumtype::Shape ShapeType>
    struct reference_element{

        //determining the order of the local dofs
        typedef layout_map<2,1,0> layout_t;
        typedef cell<Order, ShapeType> cell_t;

        static const typename basis_select<Order, BasisType, ShapeType>::type hexBasis;                       // create hex basis
        //static const Basis_HDIV_HEX_In_FEM<double, Intrepid::FieldContainer<double> > hexBasis(2, POINTTYPE_EQUISPACED);

        // choices for Gauss-Lobatto:
        // POINTTYPE_EQUISPACED = 0,
        // POINTTYPE_SPECTRAL,
        // POINTTYPE_SPECTRAL_OPEN,
        // POINTTYPE_WARPBLEND

        static const uint_t order=Order;
        static const enumtype::Shape shape=ShapeType;
        static const constexpr int spaceDim=shape_property<ShapeType>::dimension;
        static const /*constexpr*/ int numNodes;
        static const /*constexpr*/ int basisCardinality;

        //! [tensor product]
        using hypercube_t = tensor_product_element<spaceDim,order>;
        //! [tensor product]
    };

    template <ushort_t Order, enumtype::Basis BasisType, enumtype::Shape ShapeType>
    const typename basis_select<Order, BasisType, ShapeType>::type
    reference_element<Order, BasisType, ShapeType>::hexBasis;

    template <ushort_t Order, enumtype::Basis BasisType, enumtype::Shape ShapeType>
    const constexpr int reference_element<Order, BasisType, ShapeType>::spaceDim;// = cellType.getDimension();

    template <ushort_t Order, enumtype::Basis BasisType, enumtype::Shape ShapeType>
    const int reference_element<Order, BasisType, ShapeType>::numNodes = cell_t::value.getNodeCount();

    template <ushort_t Order, enumtype::Basis BasisType, enumtype::Shape ShapeType>
    const int reference_element<Order, BasisType, ShapeType>::basisCardinality = hexBasis.getCardinality();
//! [fe namespace]

}//namespace gridtools
