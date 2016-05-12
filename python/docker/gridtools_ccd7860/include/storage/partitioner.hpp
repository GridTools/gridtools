#pragma once
#include"common/defs.hpp"
#include "common/halo_descriptor.hpp"
#ifdef HAS_GCL
#include"communication/halo_exchange.hpp"
#endif
#include "cell_topology.hpp"
#include "../common/gt_math.hpp"

/**
@file
@brief Simple Partitioner Class
This file defines a simple partitioner splitting a structured cartesian grid
*/
namespace gridtools{

    struct partitioner_dummy;
    template <typename GridTopology, typename Communicator>
    class partitioner_trivial;

    template <typename Derived>
    struct space_dimensions;

    template<>
    struct space_dimensions<partitioner_dummy>{static const ushort_t value = 3;};

    template<typename TopologyType, typename Communicator>
    struct space_dimensions<partitioner_trivial<TopologyType, Communicator> >{static const ushort_t value = TopologyType::space_dimensions;};

template <typename Derived>
class partitioner {

public:

    enum Flag{UP=1, LOW=
#ifdef CXX11_ENABLED
              gt_pow<space_dimensions<Derived>::value>::apply(2)
#else
              8 // 2^3, 3D topology
#endif
    };

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


    /**@brief dummy partitioner, must be empty (only static data)

       used in the case in which no partitioner is needed
     */
    class partitioner_dummy : public partitioner<partitioner_dummy> {

    public:
        typedef partitioner< partitioner_dummy > super;

        static int boundary() {return 64+32+16+8+4+2+1;}
        static bool at_boundary(ushort_t const& /*coordinate*/, super::Flag const& /*flag_*/) {return true;}
        static const ushort_t space_dimensions=3;
    };

    template<typename Partitioner>
    struct is_partitioner_dummy : boost::false_type{};

    template<>
    struct is_partitioner_dummy<partitioner_dummy> : boost::true_type{};

}//namespace gridtools
