#pragma once

// [includes]
#include <Intrepid_FunctionSpaceTools.hpp>
#include <Intrepid_Types.hpp>

#include <gridtools.hpp>
#include <stencil-composition/accessor.hpp>
#include <stencil-composition/interval.hpp>

#include <boost/type_traits.hpp>
// [includes]

//#define REORDER

typedef gridtools::interval<gridtools::level<0,-1>, gridtools::level<1,-1> > x_interval;
typedef gridtools::interval<gridtools::level<0,-2>, gridtools::level<1,1> > axis;
#include "assembly.h"

namespace intrepid{
    using namespace Intrepid;
    bool test(){

        // [test]
        typedef storage_t<layout_map<0,1,2> >::storage_t::value_type value_type;
        typedef storage_t<layout_map<0,1,2,3> >::storage_t::layout layout_local_grid;
        GRIDTOOLS_STATIC_ASSERT(layout_local_grid::at_<0>::value < 3 && layout_local_grid::at_<1>::value < 3 && layout_local_grid::at_<2>::value < 3,
                                "the first three numbers in the layout_map must be a permutation of {0,1,2}. ");

        uint_t d1=6;
        uint_t d2=6;
        uint_t d3=1;

        //create the local grid
        storage_t<layout_map<0,1,2> >::storage_t local_grid_s(geo_map::basisCardinality, geo_map::spaceDim,1);
        storage_t<layout_map<0,1,2> > local_grid_i(local_grid_s, 2);
        geo_map::hexBasis.getDofCoords(local_grid_i);

        //! [reorder]
        std::vector<uint_t> permutations( fe::basisCardinality );
        std::vector<uint_t> to_reorder( fe::basisCardinality );
        //sorting the a vector containing the point coordinates with priority i->j->k, and saving the permutation
#ifdef REORDER

        for(uint_t i=0; i<fe::basisCardinality; ++i){
            to_reorder[i]=(local_grid_s(i,layout_local_grid::at_<0>::value)+2)*4+(local_grid_s(i,layout_local_grid::at_<1>::value)+2)*2+(local_grid_s(i,layout_local_grid::at_<2>::value)+2);
            permutations[i]=i;
        }

        std::sort(permutations.begin(), permutations.end(),
                  [&to_reorder](uint_t a, uint_t b){
                      return to_reorder[a]<to_reorder[b];
                  } );

        storage_t<layout_map<0,1,2> >::storage_t  local_grid_reordered_s(geo_map::basisCardinality, geo_map::spaceDim,1);
        storage_t<layout_map<0,1,2> >  local_grid_reordered_i(local_grid_reordered_s, 2);        //storage2 local_grid_reordered(local_grid_reordered_s, 2);
        uint_t D=geo_map::basisCardinality;

        //applying the permutation to the grid
        for(uint_t i=0; i<D; ++i){//few redundant loops
            {
                local_grid_reordered_s(i, 0)=local_grid_s(permutations[i],0);
                local_grid_reordered_s(i, 1)=local_grid_s(permutations[i],1);
                local_grid_reordered_s(i, 2)=local_grid_s(permutations[i],2);
            }
        }
        //! [reorder]
#else
        for(uint_t i=0; i<fe::basisCardinality; ++i){
            permutations[i]=i;
        }

#endif

        storage_t<layout_map<0,1,2> >::storage_t cub_points_s(cubature::numCubPoints, fe::spaceDim,1);
        storage_t<layout_map<0,1,2> > cub_points_i(cub_points_s, 2);

        storage_t<layout_map<0,1,2> >::storage_t cub_weights_s(cubature::numCubPoints,1,1);
        storage_t<layout_map<0,1,2> > cub_weights_i(cub_weights_s, 1);

        storage_t<layout_map<0,1,2> >::storage_t grad_at_cub_points_s(fe::basisCardinality, cubature::numCubPoints, fe::spaceDim);
        storage_t<layout_map<0,1,2> > grad_at_cub_points_i(grad_at_cub_points_s);

        storage_t<layout_map<0,1,2> >::storage_t grad_ordered(fe::basisCardinality, cubature::numCubPoints, fe::spaceDim);

        // retrieve cubature points and weights
        cubature::cub->getCubature(cub_points_i, cub_weights_i);
        //copy
        // evaluate grad operator at cubature points
        fe::hexBasis.getValues(grad_at_cub_points_i, cub_points_i, OPERATOR_GRAD);


        assembly assembly_(d1, d2, d3);

        assembly_.compute(
#ifdef REORDER
            local_grid_reordered_s
#else
            local_grid_s
#endif
            , grad_at_cub_points_s, cub_weights_s );
        // [test]

        // [reference & comparison]
        FieldContainer<double> grad_at_cub_points(fe::basisCardinality, cubature::numCubPoints, fe::spaceDim);

        for (uint_t i=0; i<fe::basisCardinality; ++i)
            for (uint_t j=0; j<cubature::numCubPoints; ++j)
                for (int_t k=0; k<fe::spaceDim; ++k)
                    grad_at_cub_points(i,j,k)=grad_at_cub_points_s(i,j,k);

        FieldContainer<double> cub_weights(cubature::numCubPoints);

        for(uint_t i=0; i<cubature::numCubPoints; ++i)
        {
            cub_weights(i)=cub_weights_i(i);
        }

        storage_t<layout_map<0,1,2> >::storage_t grid_s(d1*d2*d3, geo_map::basisCardinality, 3);
        wrap_pointer<float_type> ptr_(assembly_.get_grid().fields()[0].get(), assembly_.get_grid().size(), true);
        grid_s.set(ptr_);
        storage_t<layout_map<0,1,2> > grid_i(grid_s);

//         // wrap_pointer<float_type> ptr_(assembly_.get_grid().fields()[0].get(), assembly_.get_grid().size(), true);


        storage_t<layout_map<0,1,2,3> >::storage_t jac_s ((d1*d2*d3), cubature::numCubPoints, geo_map::spaceDim, geo_map::spaceDim);
        jac_s.initialize(0.);
        storage_t<layout_map<0,1,2,3> > jac_i (jac_s);
        FieldContainer<double> jac((d1*d2*d3), cubature::numCubPoints, geo_map::spaceDim, geo_map::spaceDim);

        for (int i =0; i< d1; ++i)
            for (int j =0; j< d2; ++j)
                for (int k =0; k< d3; ++k)
                {
                    for (int q=0; q<geo_map::basisCardinality; ++q)
                    {
                        for (int d=0; d<3; ++d)
                        {
                            assert(assembly_.get_grid()(i, j, k, q, d) == grid_i(i*d2*d3+j*d3+k, q, d));
                        }
                    }
                }

        CellTools<double>::setJacobian(jac_i, cub_points_i, grid_i, geo_map::cellType);
        CellTools<double>::setJacobian(jac, cub_points_i, grid_i, geo_map::cellType);

        auto epsilon=1e-15;

#ifndef REORDER
        for (int i =0; i< d1; ++i)
            for (int j =0; j< d2; ++j)
                for (int k =0; k< d3; ++k)
                    for (int q=0; q<cubature::numCubPoints; ++q)
                    {
                        for (int dimx=0; dimx<geo_map::spaceDim; ++dimx)
                            for (int dimy=0; dimy<geo_map::spaceDim; ++dimy)
                        {
                            if(assembly_.get_jac()(i, j, k, q, dimx, dimy) > epsilon+ jac(i*d2*d3+j*d3+k, q, dimx, dimy)/*weighted_measure(i*d2*d3+j*d3+k, q)*/
                            ||
                               assembly_.get_jac()(i, j, k, q, dimx, dimy) +epsilon < jac(i*d2*d3+j*d3+k, q, dimx, dimy)// weighted_measure(i*d2*d3+j*d3+k, q)
                            ){
                                std::cout<<"error in i="<<i<<" j="<<j<<" k="<<k<<" q="<<q<<" "<<dimx<<" "<<dimy<<": "
                                         <<assembly_.get_jac()(i, j, k, q, dimx, dimy)<<" != "
                                     <<jac_i(i*d3*d2+j*d3+k, q, dimx, dimy)// weighted_measure(i*d2*d3+j*d3+k, q)
                                     <<std::endl;
                                assert(false);

                            // // assert(assembly_.get_jac()(i, j, k, q, dimx, dimy) == jac(i*d2*d3+j*d3+k, q, dimx, dimy));
                            }
                        }
                    }
#endif

        storage_t<layout_map<0,1,2> >::storage_t jac_det_s ((d1*d2*d3), cubature::numCubPoints, 1);
        storage_t<layout_map<0,1,2> > jac_det_i (jac_det_s,2);
        FieldContainer<double> jac_det((d1*d2*d3), cubature::numCubPoints);

        storage_t<layout_map<0,1,2> >::storage_t weighted_measure_s ((d1*d2*d3), cubature::numCubPoints, 1);
        storage_t<layout_map<0,1,2> > weighted_measure_i (weighted_measure_s,2);
        FieldContainer<double> weighted_measure((d1*d2*d3), cubature::numCubPoints);

        CellTools<double>::setJacobianDet(jac_det_i, jac_i);
        CellTools<double>::setJacobianDet(jac_det, jac);
        FunctionSpaceTools::computeCellMeasure<double>(weighted_measure,                      // compute weighted cell measure
                                                        jac_det,
                                                        cub_weights);

#ifndef REORDER
        for (int i =0; i< d1; ++i)
        {
            for (int j =0; j< d2; ++j)
            {
                for (int k =0; k< d3; ++k)
                {
                    for (int q=0; q<cubature::numCubPoints; ++q)
                    {
                        if(assembly_.get_jac_det()(i, j, k, q) > epsilon+ jac_det_i(i*d2*d3+j*d3+k, q)/*weighted_measure(i*d2*d3+j*d3+k, q)*/
                            ||
                            assembly_.get_jac_det()(i, j, k, q) +epsilon < jac_det_i(i*d2*d3+j*d3+k, q)// weighted_measure(i*d2*d3+j*d3+k, q)
                            ){
                            std::cout<<"error in i="<<i<<" j="<<j<<" k="<<k<<" q="<<q<<": "
                                     <<assembly_.get_jac_det()(i, j, k, q)<<" != "
                                     <<jac_det_i(i*d2*d3+j*d3+k, q)// weighted_measure(i*d2*d3+j*d3+k, q)
                                     <<std::endl;
                            assert(false);
                        }
                    }
                }
            }
        }
#endif

        storage_t<layout_map<0,1,2,3> >::storage_t jac_inv_s ((d1*d2*d3), cubature::numCubPoints, geo_map::spaceDim, geo_map::spaceDim);
        storage_t<layout_map<0,1,2,3> > jac_inv_i (jac_inv_s);
        FieldContainer<double> jac_inv((d1*d2*d3), cubature::numCubPoints, 3, 3);
        CellTools<double>::setJacobianInv(jac_inv_i, jac_i);
        CellTools<double>::setJacobianInv(jac_inv, jac);

#ifndef REORDER
        for (int i =0; i< d1; ++i)
        {
            for (int j =0; j< d2; ++j)
            {
                for (int k =0; k< d3; ++k)
                {
                    for (int q=0; q<cubature::numCubPoints; ++q)
                    {
                        for (int dimx=0; dimx<3; ++dimx)
                            for (int dimy=0; dimy<3; ++dimy)
                        {
                            if(assembly_.get_jac_inv()(i, j, k, q, dimy, dimx) > epsilon+ jac_inv_i(i*d2*d3+j*d3+k, q, dimx, dimy)/*weighted_measure(i*d2*d3+j*d3+k, q)*/
                            ||
                               assembly_.get_jac_inv()(i, j, k, q, dimy, dimx) +epsilon < jac_inv_i(i*d2*d3+j*d3+k, q, dimx, dimy)// weighted_measure(i*d2*d3+j*d3+k, q)
                            ){
                                std::cout<<"error in i="<<i<<" j="<<j<<" k="<<k<<" q="<<q<<" dimx="<<dimx<<" dimy="<<dimy<<": "
                                     <<assembly_.get_jac_inv()(i, j, k, q, dimy, dimx)<<" != "
                                     <<jac_inv_i(i*d2*d3+j*d3+k, q, dimx, dimy)// weighted_measure(i*d2*d3+j*d3+k, q)
                                     <<std::endl;
                                assert(false);
                            }
                        }
                    }
                }
            }
        }
#endif


        FieldContainer<double> transformed_grad_at_cub_points((d1*d2*d3), fe::basisCardinality, cubature::numCubPoints, fe::spaceDim);
        transformed_grad_at_cub_points.initialize(1.);

        FieldContainer<double> weighted_transformed_grad_at_cub_points((d1*d2*d3), fe::basisCardinality, cubature::numCubPoints, fe::spaceDim);
        weighted_transformed_grad_at_cub_points.initialize(1.);

        FieldContainer<double> stiffness_matrices((d1*d2*d3), fe::basisCardinality, fe::basisCardinality);

        FunctionSpaceTools::HGRADtransformGRAD<double>(transformed_grad_at_cub_points,        // transform reference gradients into physical space
                                                       jac_inv,
                                                       grad_at_cub_points);


        FunctionSpaceTools::multiplyMeasure<double>(weighted_transformed_grad_at_cub_points,  // multiply with weighted measure
                                                    weighted_measure,
                                                    transformed_grad_at_cub_points);

        FunctionSpaceTools::integrate<double>(stiffness_matrices,                             // compute stiffness matrices
                                              transformed_grad_at_cub_points,
                                              weighted_transformed_grad_at_cub_points,
                                              Intrepid::COMP_CPP);

        epsilon=1e-10;
        for (int i =0; i< d1; ++i)
        {
            for (int j =0; j< d2; ++j)
            {
                for (int k =0; k< d3; ++k)
                {
                    for (int P_i =0; P_i< fe::basisCardinality; ++P_i)
                        for (int Q_i =0; Q_i< fe::basisCardinality; ++Q_i)
                        {
                            if(assembly_.get_result()(i, j, k, P_i, Q_i) > epsilon+ stiffness_matrices(i*d2*d3+j*d3+k, permutations[P_i], permutations[Q_i])/*weighted_measure(i*d2*d3+j*d3+k, q)*/
                               ||
                               assembly_.get_result()(i, j, k, P_i, Q_i) +epsilon < stiffness_matrices(i*d2*d3+j*d3+k, permutations[P_i], permutations[Q_i])// weighted_measure(i*d2*d3+j*d3+k, q)
                                ){
                                std::cout<<"error in i="<<i<<" j="<<j<<" k="<<k<<" P_i="<<P_i<<" Q_i="<<Q_i<<": "
                                         <<assembly_.get_result()(i, j, k, P_i, Q_i)<<" != "
                                         <<stiffness_matrices(i*d2*d3+j*d3+k, permutations[P_i], permutations[Q_i])
                                         <<std::endl;
                                assert(false);
                            }
                        }
                }
            }
        }
        // [reference & comparison]

        return true;
}
}
