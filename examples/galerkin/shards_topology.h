#pragma once
#include "Shards_CellTopologyData.h"
#include "Shards_CellTopology.hpp"
#include "Shards_BasicTopologies.hpp"

namespace gridtools{

//!@class ShardsTopology
/*!
  This class wraps the definitions contained in the shards::Topology and  shards::TopologyData objects, and satisfies the requirements
  for an ElementShape used for the discretization (but not for the mesh).
*/
    template<class CellTopologyType>
    class ShardsTopology
    {
    public:

        typedef CellTopologyType type;

        //it is sufficient that the template type derives from the traits defined in Shards!

        //! These values are compile-time constants. This fact enhances compiler optimization when they are used whithin loops.
        static const shards::CellTopology* S_topology;
        static const CellTopologyData* S_topologyData;
        static const uint_t S_numPoints=CellTopologyType::Traits::node_count;//shards::getCellTopologyData<CellTopologyType>()->node_count;//super::S_topologyData->node_count;
        static const uint_t S_numEdges=CellTopologyType::Traits::edge_count;
        static const uint_t S_numVertices=CellTopologyType::Traits::node_count;
        static const uint_t S_numFaces=CellTopologyType::Traits::side_count;
        static const uint_t S_nDimensions;
        static const uint_t S_numFacets=S_numFaces;
        static const uint_t S_numRidges=S_numEdges;
        static const uint_t S_numPeaks=S_numVertices;

        // static const uint_t S_numPointsPerEdge=(int)(CellTopologyType::Traits::template edge<>::topology::node_count)-2;
        // static const uint_t S_numPointsPerVertex = 1; //!< Number of points per vertex
        // //TODO generalize for other shapes!
        // static const uint_t S_numPointsPerFace=(int)(CellTopologyType::Traits::template side<>::topology::node_count)-3/*tetra*/; //!< Number of points per face
        // static const uint_t S_numPointsPerVolume = 0; //!< Number of points per volume

#if(__cplusplus==201103L)
        //The following line uses a c++11 syntax: constexpr allows the use of functions inside templates, and helps checking if
        //an expresison is really available at compile time
        constexpr
#endif
        static const CellTopologyData& topologyData(){return *S_topologyData;}

        //private:
        //a constructor is called in RegionMesh TODO: avoid that, the Shards ElementShape is a singleton, or it is never constructed
        ShardsTopology( ) {
            //assert(false);
        }
    };

    template<class CellTopologyType>
    const CellTopologyData* ShardsTopology<CellTopologyType>::S_topologyData = shards::getCellTopologyData<CellTopologyType>();

    template<class CellTopologyType>
    const shards::CellTopology* ShardsTopology<CellTopologyType>::S_topology = new shards::CellTopology(ShardsTopology<CellTopologyType>::S_topologyData/*shards::getCellTopologyData<CellTopologyType>()*/);


    template<class CellTopologyType>
    const uint_t ShardsTopology<CellTopologyType>::S_nDimensions = CellTopologyType::Traits::dimension;
}//namespace gridtools
