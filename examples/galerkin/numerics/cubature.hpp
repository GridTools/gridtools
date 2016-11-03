/** \file */
#pragma once

#include <Intrepid_Cubature.hpp>
#include <Intrepid_DefaultCubatureFactory.hpp>
#include <Intrepid_CubatureTensor.hpp>

#include "cell.hpp"
#include "element_traits.hpp"

namespace gdl{

//! [quadrature]
    //default case
    template <ushort_t Order, enumtype::Shape CellType>
    struct cubature{
        using cell_t=cell<Order, CellType>;
        // set cubature degree, e.g. 2
        static const int cubDegree = Order;
        static const enumtype::Shape shape = CellType;
        // create cubature factory
        static Intrepid::DefaultCubatureFactory<gt::float_type,  Intrepid::FieldContainer<gt::float_type> >
        cubFactory(){

            static Intrepid::DefaultCubatureFactory<gt::float_type,  Intrepid::FieldContainer<gt::float_type> > cub_factory;
            return cub_factory;
        }
        // create default cubature
        static Teuchos::RCP<Intrepid::Cubature<gt::float_type, Intrepid::FieldContainer<gt::float_type> > >
        cub(){
            static const auto cub_=cubFactory().create(cell_t::value, cubature<cubDegree, CellType>::cubDegree);
            return cub_;
        }
        // static Teuchos::RCP<Intrepid::Cubature<gt::float_type, Intrepid::FieldContainer<gt::float_type> > > cub;
        // retrieve number of cubature points
        static // constexpr
        int numCubPoints(){
            return cub()->getNumPoints();
        }
    };


    // template <ushort_t Order, enumtype::Shape  CellType>
    // Intrepid::DefaultCubatureFactory<gt::float_type,  Intrepid::FieldContainer<gt::float_type> > cubature<Order, CellType>::cubFactory;

    // template <ushort_t Order, enumtype::Shape CellType>
    // const Teuchos::RCP<Intrepid::Cubature<gt::float_type, Intrepid::FieldContainer<gt::float_type> > > cubature<Order, CellType>::
    // cub = cubature<Order, CellType>::cubFactory.create(cell_t::value, cubature<Order, CellType>::cubDegree);

    // template <ushort_t Order, enumtype::Shape CellType>
    // const int  cubature<Order, CellType>::numCubPoints = cubature<Order, CellType>::cub->getNumPoints();
//! [quadrature]

    template <typename Cubature>
    struct boundary_cubature{

        static const Teuchos::RCP<Intrepid::Cubature<gt::float_type, Intrepid::FieldContainer<gt::float_type> > > bd_cub;
        static const enumtype::Shape shape = shape_property<Cubature::shape>::boundary;


        /** @brief returns the quadrature points of the face embedded in the origilan cell.

            performs a lift of the quadrature points onto the higher dimensional form
            (e.h. square to hexahedron).
            \param refGaussPoints output array of gauss points
            \param paramaussPoints input array of gauss points on the reference subcell (e.g. square)
            \param subcellOrd number of the specific face (e.g. square) of the current cell
            (e.g. hexahedron)
         */
        template <typename Out, typename In>
        static void lift( Out& refGaussPoints, In const& paramGaussPoints, ushort_t subcellOrd ){
            Intrepid::CellTools<float_type>::mapToReferenceSubcell(
                refGaussPoints,
                paramGaussPoints,
                shape_property<shape>::dimension,
                subcellOrd,
                //parent cell
                cell<Cubature::cubDegree, Cubature::shape>::value);
        }
    };

    template <typename Cubature>
    const Teuchos::RCP<Intrepid::Cubature<gt::float_type, Intrepid::FieldContainer<gt::float_type> > > boundary_cubature<Cubature>::
    bd_cub = Cubature::cubFactory.create(
        cell<Cubature::cubDegree, shape_property<Cubature::shape>::boundary>::value
        , Cubature::cubDegree);

}//namespace gdl
