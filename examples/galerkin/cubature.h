/** \file */
#pragma once

#include <Intrepid_Cubature.hpp>
#include <Intrepid_DefaultCubatureFactory.hpp>

namespace gridtools{

//! [quadrature]
    template <ushort_t Order, enumtype::Shape CellType>
    struct cubature{
        using cell_t=cell<Order, CellType>;
        // set cubature degree, e.g. 2
        static const int cubDegree = Order;
        // create cubature factory
        static Intrepid::DefaultCubatureFactory<double,  Intrepid::FieldContainer<double> > cubFactory;
        // create default cubature
        static const Teuchos::RCP<Intrepid::Cubature<double, Intrepid::FieldContainer<double> > > cub;
        // retrieve number of cubature points
        static const int numCubPoints;
    };

    template <ushort_t Order, enumtype::Shape  CellType>
    Intrepid::DefaultCubatureFactory<double,  Intrepid::FieldContainer<double> > cubature<Order, CellType>::cubFactory;

    template <ushort_t Order, enumtype::Shape CellType>
    const Teuchos::RCP<Intrepid::Cubature<double, Intrepid::FieldContainer<double> > > cubature<Order, CellType>::
    cub = cubature<Order, CellType>::cubFactory.create(cell_t::value, cubature<Order, CellType>::cubDegree);

    template <ushort_t Order, enumtype::Shape CellType>
    const int  cubature<Order, CellType>::numCubPoints = cubature<Order, CellType>::cub->getNumPoints();
//! [quadrature]

}//namespace gridtools
