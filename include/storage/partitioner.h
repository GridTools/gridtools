#pragma once
#include<common/defs.h>
#include <common/halo_descriptor.h>
#ifdef HAS_GCL
#include<communication/halo_exchange.h>
#endif
#include "cell_topology.h"

/**
@file
@brief Simple Partitioner Class
This file defines a simple partitioner splitting a structured cartesian grid
*/
namespace gridtools{

template <typename Derived>
class partitioner {

public:
    /**@brief constructor
       suppose we are using an MPI cartesian communicator:
       then we have a coordinates (e.g. the local i,j,k identifying a processor id) and dimensions (e.g. IxJxK)
    */
    partitioner(){}

    /**@brief computes the lower and upprt index of the local interval
       \param component the dimension being partitioned
       \param size the total size of the quantity being partitioned

       The bounds must be inclusive of the halo region
    */
#ifdef CXX11_ENABLED
    template<typename ... UInt>
    void compute_bounds(uint_t* dims,
                        halo_descriptor * coordinates,
                        halo_descriptor * coordinates_gcl,
                        int_t* low_bound,
                        int_t* up_bound,
                        UInt const& ... original_sizes
        ) const
        {
            static_cast<Derived*>(this)->compute_bounds(dims, coordinates, coordinates_gcl, low_bound, up_bound, original_sizes...);
        }
#else
    void compute_bounds(uint_t* dims,
                        halo_descriptor * coordinates,
                        halo_descriptor * coordinates_gcl,
                        int_t* low_bound,
                        int_t* up_bound,
                        uint_t const& d1,
                        uint_t const& d2,
                        uint_t const& d3
        ) const
        {
            static_cast<Derived*>(this)->compute_bounds(dims, coordinates, coordinates_gcl, low_bound, up_bound, d1, d2, d3);
        }
#endif
};

}//namespace gridtools
