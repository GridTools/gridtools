#pragma once

// [includes]
#include <Intrepid_FunctionSpaceTools.hpp>
#include <Intrepid_Types.hpp>

#include <gridtools.hpp>
#include <stencil-composition/accessor.hpp>
#include <stencil-composition/interval.hpp>

#include <boost/type_traits.hpp>
#include "basis_functions.h"
#include "cubature.h"
// [includes]

#define REORDER

// #include "assembly.h"

namespace gridtools{
namespace intrepid{

    template <typename FE, typename Cub>
    struct discretization{
        using fe=FE;
        using cub=Cub;

        // [test]
        typedef storage_t<layout_map<0,1,2> >::value_type value_type;
        GRIDTOOLS_STATIC_ASSERT(fe::layout_t::template at_<0>::value < 3 && fe::layout_t::template at_<1>::value < 3 && fe::layout_t::template at_<2>::value < 3,
                                "the first three numbers in the layout_map must be a permutation of {0,1,2}. ");
        typedef storage_t<layout_map<0,1,2> > weights_storage_t;
        typedef storage_t<layout_map<0,1,2> > grad_storage_t;
        typedef storage_t<layout_map<0,1,2> > basis_function_storage_t;

        storage_t<layout_map<0,1,2> > m_cub_points_s;
        weights_storage_t m_cub_weights_s;
        std::unique_ptr<grad_storage_t> m_grad_at_cub_points_s;
        std::unique_ptr<basis_function_storage_t> m_phi_at_cub_points_s;

        weights_storage_t // const
        & cub_weights()// const
            {return m_cub_weights_s;}
        grad_storage_t // const
        & local_gradient()// const
            {return *m_grad_at_cub_points_s;}

        basis_function_storage_t // const
        & basis_function()// const
            {return *m_phi_at_cub_points_s;}

        discretization() :
            m_cub_points_s(cub::numCubPoints, fe::spaceDim,1)
            , m_cub_weights_s(cub::numCubPoints,1,1)
            , m_grad_at_cub_points_s()
            , m_phi_at_cub_points_s()
            {

            }

        void compute(Intrepid::EOperator const& operator_){

                // storage_t<layout_map<0,1,2> > cub_points_i(m_cub_points_s, 2);
            Intrepid::FieldContainer<double> cub_points_i(cub::numCubPoints, fe::spaceDim);

                // storage_t<layout_map<0,1,2> > cub_weights_i(m_cub_weights_s, 1);
            Intrepid::FieldContainer<double> cub_weights_i(cub::numCubPoints);

                //copy the values
                for (uint_t q=0; q<cub::numCubPoints; ++q)
                {
                    m_cub_weights_s(q,0,0)=cub_weights_i(q);
                    for (uint_t j=0; j<fe::spaceDim; ++j)
                        m_cub_points_s(q,j,0)=cub_points_i(q,j);
                }

                // storage_t<layout_map<0,1,2> > grad_at_cub_points_i(m_grad_at_cub_points_s);
                Intrepid::FieldContainer<double> grad_at_cub_points_i(fe::basisCardinality, cub::numCubPoints, fe::spaceDim);

                // retrieve cub points and weights
                cub::cub->getCubature(cub_points_i, cub_weights_i);

                switch (operator_){
                case Intrepid::OPERATOR_GRAD :
                {
                    m_grad_at_cub_points_s=std::unique_ptr<grad_storage_t>(new grad_storage_t(fe::basisCardinality, cub::numCubPoints, fe::spaceDim));

                    // evaluate grad operator at cub points
                    fe::hexBasis.getValues(grad_at_cub_points_i, cub_points_i, Intrepid::OPERATOR_GRAD);

                    for (uint_t q=0; q<cub::numCubPoints; ++q)
                        for (uint_t j=0; j<fe::spaceDim; ++j)
                            for (uint_t i=0; i<fe::basisCardinality; ++i)
                                for (uint_t j=0; j<fe::spaceDim; ++j)
                                    (*m_grad_at_cub_points_s)(i,q,j)=grad_at_cub_points_i(i,q,j);
                    break;
                }
                case Intrepid::OPERATOR_VALUE :
                {
                    m_phi_at_cub_points_s=std::unique_ptr<basis_function_storage_t>(new basis_function_storage_t(fe::basisCardinality, cub::numCubPoints, 1));
                    Intrepid::FieldContainer<double> phi_at_cub_points_i(fe::basisCardinality, cub::numCubPoints);
                    fe::hexBasis.getValues(phi_at_cub_points_i, cub_points_i, Intrepid::OPERATOR_VALUE);
                    //copy the values
                    for (uint_t q=0; q<cub::numCubPoints; ++q)
                        for (uint_t j=0; j<fe::spaceDim; ++j)
                            for (uint_t i=0; i<fe::basisCardinality; ++i)
                                (*m_phi_at_cub_points_s)(i,q,0)=phi_at_cub_points_i(i,q);;
                    break;
                }
                default : assert(false);
                }

        }
    };


    template < typename GeoMap, typename Cubature >
    struct geometry : public discretization<GeoMap, Cubature>{

        using geo_map=GeoMap;
        using super=discretization<GeoMap, Cubature>;

        storage_t<layout_map<0,1,2> > m_local_grid_s;
#ifdef REORDER
        storage_t<layout_map<0,1,2> > m_local_grid_reordered_s;
#endif

        geometry() :
            //create the local grid
            m_local_grid_s(geo_map::basisCardinality, geo_map::spaceDim,1)
#ifdef REORDER
            , m_local_grid_reordered_s(geo_map::basisCardinality, geo_map::spaceDim,1)
#endif
            {
                // storage_t<layout_map<0,1,2> > local_grid_i(m_local_grid_s, 2);
                Intrepid::FieldContainer<double> local_grid_i(geo_map::basisCardinality, geo_map::spaceDim);
                geo_map::hexBasis.getDofCoords(local_grid_i);
                for (uint_t i=0; i<geo_map::basisCardinality; ++i)
                    for (uint_t j=0; j<geo_map::spaceDim; ++j)
                        m_local_grid_s(i,j,0)=local_grid_i(i,j);

                //! [reorder]
                std::vector<uint_t> permutations( geo_map::basisCardinality );
                std::vector<uint_t> to_reorder( geo_map::basisCardinality );
                //sorting the a vector containing the point coordinates with priority i->j->k, and saving the permutation
#ifdef REORDER
                // fill in the reorder vector such that the larger numbers correspond to larger strides
                for(uint_t i=0; i<geo_map::basisCardinality; ++i){
                    to_reorder[i]=(m_local_grid_s(i,geo_map::layout_t::template at_<0>::value)+2)*4+(m_local_grid_s(i,geo_map::layout_t::template at_<1>::value)+2)*2+(m_local_grid_s(i,geo_map::layout_t::template at_<2>::value)+2);
                    permutations[i]=i;
                }

                std::sort(permutations.begin(), permutations.end(),
                  [&to_reorder](uint_t a, uint_t b){
                      return to_reorder[a]<to_reorder[b];
                  } );

                // storage_t<layout_map<0,1,2> >::storage_t  local_grid_reordered_s(geo_map::basisCardinality, geo_map::spaceDim,1);
                // storage_t<layout_map<0,1,2> >  local_grid_reordered_i(m_local_grid_reordered_s, 2);
                uint_t D=geo_map::basisCardinality;

                //applying the permutation to the grid
                for(uint_t i=0; i<D; ++i){//few redundant loops
                    {
                        m_local_grid_reordered_s(i, 0, 0)=m_local_grid_s(permutations[i],0,0);
                        m_local_grid_reordered_s(i, 1, 0)=m_local_grid_s(permutations[i],1,0);
                        m_local_grid_reordered_s(i, 2, 0)=m_local_grid_s(permutations[i],2,0);
                    }
                }
                //! [reorder]
#endif
                super::compute(Intrepid::OPERATOR_GRAD);
            }

        storage_t<layout_map<0,1,2> > const& grid(){return m_local_grid_s;}

    };

}//namespace intrepid
}//namespace gridtools
