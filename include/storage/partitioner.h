#pragma once
#include<common/defs.h>

/**
@file
@brief Simple Partitioner Class
This file defines a simple partitioner splitting a structured cartesian grid
*/

template <typename Derived>
class partitioner {

public:
    /**@brief constructor
       suppose we are using an MPI cartesian communicator:
       then we have a coordinates (e.g. the local i,j,k identifying a processor id) and dimensions (e.g. IxJxK)
    */
    partitioner(){}

    // virtual uint_t size(ushort_t const& component) const=0;
    // virtual uint_t* sizes()=0;
    // virtual uint_t compute_bounds(ushort_t const& component, uint_t const&size)=0;

};
