#pragma once

/**
   @file traits classes for the various grid topologies
*/

namespace gridtools{
    namespace topology{

        /**@brief cartesian topology

           \tparam the local layout map, i.e. defining the order of the dimensions
         */
        template<typename Layout>
        struct cartesian{};
    }//namespace topology

    using namespace topology;

    template<typename TopologyType>
    class cell_topology{};

    template <typename Layout>
    class cell_topology<cartesian<Layout> >
    {
    public:
        static const ushort_t space_dimensions=Layout::length;
    };
}//namespace gridtools
