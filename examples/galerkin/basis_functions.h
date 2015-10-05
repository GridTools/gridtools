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

#ifdef __CUDACC__
    template<short_t ... Dims>
    using layout_t=gridtools::layout_map< Dims ... ,2,1,0 >;
#else
    template<short_t ... Dims>
    using layout_t=gridtools::layout_map< 0,1,2, Dims ... >;
#endif

    template <typename MetaData>
    using storage_t = typename gridtools::BACKEND::storage_type<float_type, MetaData >::type;
//! [storage definition]
//! [fe namespace]
    template <ushort_t Order, enumtype::Basis BasisType, enumtype::Shape ShapeType>
    struct reference_element{

        //determining the order of the local dofs
        typedef layout_map<2,1,0> layout_t;
        typedef cell<Order, ShapeType> cell_t;

        static const typename basis_select<Order, BasisType, ShapeType>::type
        hexBasis()                       // create hex basis
        {
            static const typename basis_select<Order, BasisType, ShapeType>::type hex_basis;
            return hex_basis;

        }
        //static const Basis_HDIV_HEX_In_FEM<double, Intrepid::FieldContainer<double> > hexBasis(2, POINTTYPE_EQUISPACED);

        // choices for Gauss-Lobatto:
        // POINTTYPE_EQUISPACED = 0,
        // POINTTYPE_SPECTRAL,
        // POINTTYPE_SPECTRAL_OPEN,
        // POINTTYPE_WARPBLEND

        static const enumtype::Basis basis=BasisType;
        static const uint_t order=Order;
        static const enumtype::Shape shape=ShapeType;
        static const constexpr int spaceDim=shape_property<ShapeType>::dimension;
        static const /*constexpr*/ int numNodes;
        static const /*constexpr*/ int basisCardinality;

        //! [tensor product]
        using hypercube_t = tensor_product_element<spaceDim,order>;
        //! [tensor product]
    };

    // template<typename T>
    // struct local_normals;

    // template< int_t  I>
    // struct local_normals<enumtype::dimension<0>{I}>{

    //     GT_FUNCTION
    //     constexpr static array<ushort_t, 3> value(){ return {I,0,0}; }
    // };

    // template< int_t J >
    // GT_FUNCTION
    // struct local_normals<enumtype::dimension<1>{J}>{
    //     constexpr static const array<ushort_t, 3> value={0,J,0};
    // };

    // template< int_t K >
    // GT_FUNCTION
    // struct local_normals<enumtype::dimension<2>{K}>{
    //     constexpr static const array<ushort_t, 3> value={0,0,K};
    // };


    template <ushort_t Order, enumtype::Basis BasisType, enumtype::Shape ShapeType>
    const constexpr int reference_element<Order, BasisType, ShapeType>::spaceDim;// = cellType.getDimension();

    template <ushort_t Order, enumtype::Basis BasisType, enumtype::Shape ShapeType>
    const int reference_element<Order, BasisType, ShapeType>::numNodes = cell_t::value.getNodeCount();

    template <ushort_t Order, enumtype::Basis BasisType, enumtype::Shape ShapeType>
    const int reference_element<Order, BasisType, ShapeType>::basisCardinality = hexBasis().getCardinality();
//! [fe namespace]


    /** compute ricci symbol*/
    constexpr ushort_t ricci(ushort_t const& i_, ushort_t const& j_){
        return i_<j_ ? 1 : i_==j_ ? 0 : -1;
    }
    constexpr ushort_t ricci(ushort_t const& i_, ushort_t const& j_, ushort_t const& k_){
        return ricci(i_,j_)*ricci(j_,k_)*ricci(i_,k_);
    }

    /**compute vector product*/
    template <typename T>
    constexpr array<T, 3> vec_product(array<T, 3> const& v1, array<T, 3> const& v2)
    {
        return array<T,3>{ricci(0,1,2)*v1[1]*v2[2]+ricci(0,2,1)*v1[2]*v2[1],
                ricci(1,0,2)*v1[0]*v2[2]+ricci(1,2,0)*v1[2]*v2[0],
                ricci(2,0,1)*v1[0]*v2[1]+ricci(2,1,0)*v1[1]*v2[0],
                };
    }

}//namespace gridtools
