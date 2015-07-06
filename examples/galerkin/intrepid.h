#pragma once

#include <gridtools.h>
#include <stencil-composition/accessor.h>

#include <boost/type_traits.hpp>

#include <Intrepid_Cubature.hpp>
// #include <Intrepid_FieldContainer.hpp>
#include <Intrepid_DefaultCubatureFactory.hpp>

//just some of the possible discretizations
#include "assembly_reference.h"
#include <stencil-composition/interval.h>


typedef gridtools::interval<gridtools::level<0,-1>, gridtools::level<1,-1> > x_interval;
typedef gridtools::interval<gridtools::level<0,-2>, gridtools::level<1,1> > axis;
#include "assembly.h"

namespace intrepid{

    using namespace Intrepid;
    bool test(){

        typedef storage3::storage_t::value_type value_type;
        typedef gridtools::layout_map<0,1,2> layout_local_grid;

        uint_t d1=6;
        uint_t d2=6;
        uint_t d3=1;

        // using tensor_product_element<3,1> = hypercube_;
        //create the grid
        storage3::storage_t local_grid_s( geo_map::basisCardinality, geo_map::spaceDim, 1);
        storage3 local_grid(local_grid_s, 2);
        geo_map::hexBasis.getDofCoords(local_grid);

        // storage3::storage_t ordered_local_grid_s( geo_map::basisCardinality, geo_map::spaceDim, 1);
        //sort
                // std::sort(&local_grid_s(0,0), &local_grid_s(0,1),
                //   [local_grid&](value_type const& arg1, value_type const& arg2)->bool{
                //       uint_t indices1[3];
                //       local_grid.iterator_to_indices(arg2-arg1, indices, local_grid.strides());
                //       return local_grid(indices[0], indices[1], indices[2])
                //   })

        std::vector<uint_t> permutations( fe::basisCardinality );
        std::vector<uint_t> to_reorder( fe::basisCardinality );
        for(uint_t i=0; i<fe::basisCardinality; ++i){
            to_reorder[i]=(local_grid(i,layout_local_grid::at_<0>::value)+2)*4+(local_grid(i,layout_local_grid::at_<1>::value)+2)*2+(local_grid(i,layout_local_grid::at_<2>::value)+2);
            permutations[i]=i;
        }
        std::sort(permutations.begin(), permutations.end(),
                  [&to_reorder](uint_t a, uint_t b){
                      return to_reorder[a]<to_reorder[b];
                  } );


        storage4::storage_t local_grid_reordered_s(2,2,2,3);
        storage4 local_grid_reordered(local_grid_reordered_s, 4);
        // for(uint_t i=0; i<fe::basisCardinality; ++i){
        uint_t dim=3;
        uint_t D=2;//get this from the order of the basis
        for(uint_t i=0; i<D; ++i){//few redundant loops
            for(uint_t j=0; j<D; ++j){
                for(uint_t k=0; k<D; ++k){
                    {
                        local_grid_reordered(i, j, k, 0)=local_grid(permutations[i*D*D+j*D+k],0);
                        local_grid_reordered(i, j, k, 1)=local_grid(permutations[i*D*D+j*D+k],1);
                        local_grid_reordered(i, j, k, 2)=local_grid(permutations[i*D*D+j*D+k],2);
                        std::cout<<"perm "<<i*(D*D)+j*D+k<<": "<<permutations[i]<<std::endl;
                    }
                }
            }
        }

        storage3::storage_t cub_points_s(fe::numCubPoints, fe::spaceDim, 1);
        storage3::storage_t cub_weights_s(fe::numCubPoints, 1, 1);
        storage3 cub_points(cub_points_s, 2);
        storage3 cub_weights(cub_weights_s, 1);
        storage3::storage_t grad_at_cub_points_s(fe::basisCardinality, fe::numCubPoints, fe::spaceDim);
        storage3 grad_at_cub_points(grad_at_cub_points_s);

        // retrieve cubature points and weights
        fe::myCub->getCubature(cub_points, cub_weights);
        // evaluate grad operator at cubature points
        fe::hexBasis.getValues(grad_at_cub_points, cub_points, OPERATOR_GRAD);

        assembly assembly_(d1, d2, d3);
        assembly_.compute(local_grid_reordered.get_storage(), grad_at_cub_points.get_storage() );

        // reference & comparison

        typedef gridtools::layout_map<0,1,2> layout_grid_t;
        typedef gridtools::layout_map<0,1,2,3> layout_jacobian_t;
        typedef intrepid_storage<gridtools::BACKEND::storage_type<float_type, layout_jacobian_t >::type > jacobian_type;
        typedef intrepid_storage<gridtools::BACKEND::storage_type<float_type, layout_grid_t >::type > grid_type;

        wrap_pointer<float_type> ptr_(assembly_.get_grid().fields()[0].get(), assembly_.get_grid().size(), true);

        jacobian_type::storage_t jac_s((d1*d2*d3), fe::numCubPoints, 3, 3);
        grid_type::storage_t grid_s(d1*d2*d3, geo_map::basisCardinality, 3);
        grid_s.set(ptr_);

        jacobian_type jac(jac_s);
        grid_type grid(grid_s);

        for (int i =0; i< d1; ++i)
            for (int j =0; j< d2; ++j)
                for (int k =0; k< d3; ++k)
                {
                    for (int q=0; q<geo_map::basisCardinality; ++q)
                    {
                        for (int d=0; d<3; ++d)
                        {
                            assert(assembly_.get_grid()(i, j, k, q, d) == grid.get_storage()(i*d2*d3+j*d3+k, q, d));
                        }
                    }
                }


        CellTools<double>::setJacobian(jac, cub_points, grid, geo_map::cellType);

        for (int i =0; i< d1; ++i)
        {
            for (int j =0; j< d2; ++j)
            {
                for (int k =0; k< d3; ++k)
                {
                    for (int q=0; q<fe::numCubPoints; ++q)
                    {
                        for (int dimx=0; dimx<3; ++dimx)
                            for (int dimy=0; dimy<3; ++dimy)
                        {
                            assert(assembly_.get_jac()(i, j, k, q, dimx, dimy) == jac.get_storage()(i*d2*d3+j*d3+k, q, dimx, dimy));
                        }
                    }
                }
            }
        }

        typedef gridtools::layout_map<0,1> layout_jac_det_t;
        typedef intrepid_storage<gridtools::BACKEND::storage_type<float_type, layout_jac_det_t >::type > jacobian_det_type;
         jacobian_det_type::storage_t jac_det_s((d1*d2*d3), fe::numCubPoints);
         jacobian_det_type jac_det(jac_det_s);

        CellTools<double>::setJacobianDet(jac_det, jac);

        auto epsilon=1e-15;

        for (int i =0; i< d1; ++i)
        {
            for (int j =0; j< d2; ++j)
            {
                for (int k =0; k< d3; ++k)
                {
                    for (int q=0; q<fe::numCubPoints; ++q)
                    {
                        if(assembly_.get_jac_det()(i, j, k, q) > epsilon+ jac_det.get_storage()(i*d2*d3+j*d3+k, q)
                            ||
                            assembly_.get_jac_det()(i, j, k, q) +epsilon < jac_det.get_storage()(i*d2*d3+j*d3+k, q)
                            ){
                            std::cout<<"error in i="<<i<<" j="<<j<<" k="<<k<<" q="<<q<<": "
                                     <<assembly_.get_jac_det()(i, j, k, q)<<" != "
                                     <<jac_det.get_storage()(i*d2*d3+j*d3+k, q)<<std::endl;
                            assert(false);
                        }
                    }
                }
            }
        }


        return true;
}
}
