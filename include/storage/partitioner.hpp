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


    /** @brief keyword to be used at the user interface level when specifyng the boundary conditions
     */
    struct up{
        template <typename Derived>
        static constexpr typename Derived::Flag value(){
            return Derived::UP;
        }
    };

    template<typename T>
    struct is_up : boost::mpl::false_{};

    template<>
    struct is_up<up> : boost::mpl::true_{};

    struct down{
        template <typename Derived>
        static constexpr typename Derived::Flag const& value(){
            return Derived::DOWN;
        }
    };

    template<typename T>
    struct is_down : boost::mpl::false_{};

    template<>
    struct is_down<down> : boost::mpl::true_{};

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
