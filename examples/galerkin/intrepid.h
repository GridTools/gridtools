#pragma once

// [includes]
#include <Intrepid_FunctionSpaceTools.hpp>
#include <Intrepid_Types.hpp>

#include <gridtools.hpp>
#include <stencil-composition/accessor.hpp>
#include <stencil-composition/interval.hpp>

#include <boost/type_traits.hpp>
#include "basis_functions.h"
// [includes]

#define REORDER

// #include "assembly.h"

namespace gridtools{
namespace intrepid{
    using namespace Intrepid;

    struct intrepid{
        // [test]
        typedef storage_t<layout_map<0,1,2> >::storage_t::value_type value_type;
        GRIDTOOLS_STATIC_ASSERT(fe::layout_t::at_<0>::value < 3 && fe::layout_t::at_<1>::value < 3 && fe::layout_t::at_<2>::value < 3,
                                "the first three numbers in the layout_map must be a permutation of {0,1,2}. ");
        typedef storage_t<layout_map<0,1,2> >::storage_t weights_storage_t;
        typedef storage_t<layout_map<0,1,2> >::storage_t grad_storage_t;

        storage_t<layout_map<0,1,2> >::storage_t m_local_grid_s;
#ifdef REORDER
        storage_t<layout_map<0,1,2> >::storage_t m_local_grid_reordered_s;
#endif
        storage_t<layout_map<0,1,2> >::storage_t m_cub_points_s;
        storage_t<layout_map<0,1,2> >::storage_t m_cub_weights_s;
        storage_t<layout_map<0,1,2> >::storage_t m_grad_at_cub_points_s;

        weights_storage_t // const
        & cub_weights()// const
            {return m_cub_weights_s;}
        grad_storage_t // const
        & local_gradient()// const
            {return m_grad_at_cub_points_s;}

        intrepid() :
            //create the local grid
            m_local_grid_s(geo_map::basisCardinality, geo_map::spaceDim,1)
#ifdef REORDER
            , m_local_grid_reordered_s(geo_map::basisCardinality, geo_map::spaceDim,1)
#endif
            , m_cub_points_s(cubature::numCubPoints, fe::spaceDim,1)
            , m_cub_weights_s(cubature::numCubPoints,1,1)
            , m_grad_at_cub_points_s(fe::basisCardinality, cubature::numCubPoints, fe::spaceDim){


        storage_t<layout_map<0,1,2> > local_grid_i(m_local_grid_s, 2);
        geo_map::hexBasis.getDofCoords(local_grid_i);

        //! [reorder]
        std::vector<uint_t> permutations( fe::basisCardinality );
        std::vector<uint_t> to_reorder( fe::basisCardinality );
        //sorting the a vector containing the point coordinates with priority i->j->k, and saving the permutation
#ifdef REORDER
        // fill in the reorder vector such that the larger numbers correspond to larger strides
        for(uint_t i=0; i<fe::basisCardinality; ++i){
            to_reorder[i]=(m_local_grid_s(i,fe::layout_t::at_<0>::value)+2)*4+(m_local_grid_s(i,fe::layout_t::at_<1>::value)+2)*2+(m_local_grid_s(i,fe::layout_t::at_<2>::value)+2);
            permutations[i]=i;
        }

        std::sort(permutations.begin(), permutations.end(),
                  [&to_reorder](uint_t a, uint_t b){
                      return to_reorder[a]<to_reorder[b];
                  } );

        // storage_t<layout_map<0,1,2> >::storage_t  local_grid_reordered_s(geo_map::basisCardinality, geo_map::spaceDim,1);
        storage_t<layout_map<0,1,2> >  local_grid_reordered_i(m_local_grid_reordered_s, 2);
        uint_t D=geo_map::basisCardinality;

        //applying the permutation to the grid
        for(uint_t i=0; i<D; ++i){//few redundant loops
            {
                m_local_grid_reordered_s(i, 0)=m_local_grid_s(permutations[i],0);
                m_local_grid_reordered_s(i, 1)=m_local_grid_s(permutations[i],1);
                m_local_grid_reordered_s(i, 2)=m_local_grid_s(permutations[i],2);
            }
        }
        //! [reorder]
#else
        for(uint_t i=0; i<fe::basisCardinality; ++i){
            permutations[i]=i;
        }

#endif

        storage_t<layout_map<0,1,2> > cub_points_i(m_cub_points_s, 2);

        storage_t<layout_map<0,1,2> > cub_weights_i(m_cub_weights_s, 1);

        storage_t<layout_map<0,1,2> > grad_at_cub_points_i(m_grad_at_cub_points_s);

        // retrieve cubature points and weights
        cubature::cub->getCubature(cub_points_i, cub_weights_i);
        //copy
        // evaluate grad operator at cubature points
        fe::hexBasis.getValues(grad_at_cub_points_i, cub_points_i, OPERATOR_GRAD);
    }

   };
}//namespace intrepid
}//namespace gridtools
