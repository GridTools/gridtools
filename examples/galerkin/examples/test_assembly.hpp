#pragma once
#include "../numerics/assembly.hpp"
#include <iostream>

<<<<<<< Updated upstream
namespace gdl{
=======

//namespace gridtools{
>>>>>>> Stashed changes
namespace intrepid{

    using namespace Intrepid;
    using namespace gridtools;
    using namespace gdl;
    using namespace gdl::enumtype;


    template <typename Geometry, typename FEBackend, typename MatrixType>
    bool test(assembly_base<Geometry> & assembly_base_, assembly<Geometry> & assembly_, FEBackend const& fe_backend_, MatrixType const& stiffness_){
        using fe = typename Geometry::geo_map;
        using cub= typename Geometry::cub;
        using geo_map = typename Geometry::geo_map;

        auto d1=assembly_.m_d1;
        auto d2=assembly_.m_d2;
        auto d3=assembly_.m_d3;

        // [reference & comparison]
        FieldContainer<double> grad_at_cub_points(fe::basisCardinality, cub::numCubPoints(), fe::spaceDim);

        for (uint_t i=0; i<fe::basisCardinality; ++i)
            for (uint_t j=0; j<cub::numCubPoints(); ++j)
<<<<<<< Updated upstream
                for (int_t k=0; k<fe::spaceDim; ++k)
                    grad_at_cub_points(i,j,k)=(fe_backend_.grad())(i,j,k);
=======
                for (uint_t k=0; k<fe::spaceDim; ++k)
                    grad_at_cub_points(i,j,k)=(*fe_backend_.m_grad_at_cub_points_s)(i,j,k);
>>>>>>> Stashed changes

        FieldContainer<double> cub_weights(cub::numCubPoints());

        for(uint_t i=0; i<cub::numCubPoints(); ++i)
        {
            cub_weights(i)=fe_backend_.cub_weights()(i,0,0);
        }

        FieldContainer<double> cub_points(cub::numCubPoints(), fe::spaceDim);
        for(uint_t i=0; i<cub::numCubPoints(); ++i)
        {
            for(uint_t j=0; j<fe::spaceDim; ++j)
            {
                cub_points(i,j)=fe_backend_.cub_points()(i,j,0);
            }
        }

        FieldContainer<double> grid(d1*d2*d3, geo_map::basisCardinality, 3);

        for (int i =0; i< d1; ++i)
            for (int j =0; j< d2; ++j)
                for (int k =0; k< d3; ++k)
                {
                    for (int q=0; q<geo_map::basisCardinality; ++q)
                    {
                        for (int d=0; d<3; ++d)
                        {//assign
                            grid(i*d2*d3+j*d3+k, q, d)=assembly_base_.get_grid()(i, j, k, q, d);
                        }
                    }
                }

        FieldContainer<double> jac((d1*d2*d3), cub::numCubPoints(), geo_map::spaceDim, geo_map::spaceDim);
        CellTools<double>::setJacobian(jac, cub_points, grid, geo_map::cell_t::value);

        auto epsilon=1e-15;

        for (int i =0; i< d1; ++i)
            for (int j =0; j< d2; ++j)
                for (int k =0; k< d3; ++k)
                    for (int q=0; q<cub::numCubPoints(); ++q)
                    {
                        for (int dimx=0; dimx<geo_map::spaceDim; ++dimx)
                            for (int dimy=0; dimy<geo_map::spaceDim; ++dimy)
                            {
                                std::cout<<"det val in i="<<i<<" j="<<j<<" k="<<k<<" q="<<q<<" "<<dimx<<" "<<dimy<<": "
                                         <<assembly_.get_jac()(i, j, k, q, dimx, dimy)<<" and "
                                         <<jac(i*d3*d2+j*d3+k, q, dimx, dimy)// weighted_measure(i*d2*d3+j*d3+k, q)
                                         <<std::endl;

                                if(assembly_.get_jac()(i, j, k, q, dimx, dimy) > epsilon+ jac(i*d2*d3+j*d3+k, q, dimx, dimy)/*weighted_measure(i*d2*d3+j*d3+k, q)*/
                                   ||
                                   assembly_.get_jac()(i, j, k, q, dimx, dimy) +epsilon < jac(i*d2*d3+j*d3+k, q, dimx, dimy)// weighted_measure(i*d2*d3+j*d3+k, q)
                                    ){
                                    std::cout<<"error in i="<<i<<" j="<<j<<" k="<<k<<" q="<<q<<" "<<dimx<<" "<<dimy<<": "
                                             <<assembly_.get_jac()(i, j, k, q, dimx, dimy)<<" != "
                                             <<jac(i*d3*d2+j*d3+k, q, dimx, dimy)// weighted_measure(i*d2*d3+j*d3+k, q)
                                             <<std::endl;
                                    assert(false);

                                }
                            }
                    }

        FieldContainer<double> jac_det((d1*d2*d3), cub::numCubPoints());

        // storage_t<layout_map<0,1,2> >::storage_t weighted_measure_s ((d1*d2*d3), cub::numCubPoints(), 1);
        FieldContainer<double> weighted_measure((d1*d2*d3), cub::numCubPoints());

        CellTools<double>::setJacobianDet(jac_det, jac);
        FunctionSpaceTools::computeCellMeasure<double>(weighted_measure,                      // compute weighted cell measure
                                                       jac_det,
                                                       cub_weights);

        for (int i =0; i< d1; ++i)
        {
            for (int j =0; j< d2; ++j)
            {
                for (int k =0; k< d3; ++k)
                {
                    for (int q=0; q<cub::numCubPoints(); ++q)
                    {
                        std::cout<<"jac det value  in i="<<i<<" j="<<j<<" k="<<k<<" q="<<q<<": "
                                 <<assembly_.get_jac_det()(i, j, k, q)<<" and "
                                 <<jac_det(i*d2*d3+j*d3+k, q)
                                 <<std::endl;


                    	if(assembly_.get_jac_det()(i, j, k, q) > epsilon+ jac_det(i*d2*d3+j*d3+k, q)
                           ||
                           assembly_.get_jac_det()(i, j, k, q) +epsilon < jac_det(i*d2*d3+j*d3+k, q)
                            ){
                            std::cout<<"error in i="<<i<<" j="<<j<<" k="<<k<<" q="<<q<<": "
                                     <<assembly_.get_jac_det()(i, j, k, q)<<" != "
                                     <<jac_det(i*d2*d3+j*d3+k, q)
                                     <<std::endl;
                            assert(false);
                        }
                    }
                }
            }
        }

        FieldContainer<double> jac_inv((d1*d2*d3), cub::numCubPoints(), geo_map::spaceDim, geo_map::spaceDim);
        CellTools<double>::setJacobianInv(jac_inv, jac);

        for (int i =0; i< d1; ++i)
        {
            for (int j =0; j< d2; ++j)
            {
                for (int k =0; k< d3; ++k)
                {
                    for (int q=0; q<cub::numCubPoints(); ++q)
                    {
                        for (int dimx=0; dimx<fe::spaceDim; ++dimx)
                            for (int dimy=0; dimy<fe::spaceDim; ++dimy)
                            {
                                std::cout<<"inv jac value in i="<<i<<" j="<<j<<" k="<<k<<" q="<<q<<" dimx="<<dimx<<" dimy="<<dimy<<": "
                                                                         <<assembly_.get_jac_inv()(i, j, k, q, dimy, dimx)<<" and "
                                                                         <<jac_inv(i*d2*d3+j*d3+k, q, dimx, dimy)// weighted_measure(i*d2*d3+j*d3+k, q)
                                                                         <<std::endl;


                                if(assembly_.get_jac_inv()(i, j, k, q, dimy, dimx) > epsilon+ jac_inv(i*d2*d3+j*d3+k, q, dimx, dimy)/*weighted_measure(i*d2*d3+j*d3+k, q)*/
                                   ||
                                   assembly_.get_jac_inv()(i, j, k, q, dimy, dimx) +epsilon < jac_inv(i*d2*d3+j*d3+k, q, dimx, dimy)// weighted_measure(i*d2*d3+j*d3+k, q)
                                    ){
                                    std::cout<<"error in i="<<i<<" j="<<j<<" k="<<k<<" q="<<q<<" dimx="<<dimx<<" dimy="<<dimy<<": "
                                             <<assembly_.get_jac_inv()(i, j, k, q, dimy, dimx)<<" != "
                                             <<jac_inv(i*d2*d3+j*d3+k, q, dimx, dimy)// weighted_measure(i*d2*d3+j*d3+k, q)
                                             <<std::endl;
                                    assert(false);
                                }
                            }
                    }
                }
            }
        }


        FieldContainer<double> transformed_grad_at_cub_points((d1*d2*d3), fe::basisCardinality, cub::numCubPoints(), fe::spaceDim);
        transformed_grad_at_cub_points.initialize(1.);

        FieldContainer<double> weighted_transformed_grad_at_cub_points((d1*d2*d3), fe::basisCardinality, cub::numCubPoints(), fe::spaceDim);
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

//#ifndef REORDER
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
                            std::cout<<"Value in i="<<i<<" j="<<j<<" k="<<k<<" P_i="<<P_i<<" Q_i="<<Q_i<<": "
                                     <<stiffness_(i, j, k, P_i, Q_i)<<" and "
                                     <<stiffness_matrices(i*d2*d3+j*d3+k, P_i, Q_i)
                                     <<std::endl;
                        	if(stiffness_(i, j, k, P_i, Q_i) > epsilon+ stiffness_matrices(i*d2*d3+j*d3+k, P_i, Q_i)
                               ||
                               stiffness_(i, j, k, P_i, Q_i) +epsilon < stiffness_matrices(i*d2*d3+j*d3+k, P_i, Q_i)
                                ){
                                std::cout<<"error in i="<<i<<" j="<<j<<" k="<<k<<" P_i="<<P_i<<" Q_i="<<Q_i<<": "
                                         <<stiffness_(i, j, k, P_i, Q_i)<<" != "
                                         <<stiffness_matrices(i*d2*d3+j*d3+k, P_i, Q_i)
                                         <<std::endl;
                                assert(false);
                            }
                        }
                }
            }
        }
//#endif
        // [reference & comparison]

        return true;
    }


    template <typename Geometry, typename FEBackend, typename MatrixType>
        bool test_mass(assembly_base<Geometry> & assembly_base_, assembly<Geometry> & assembly_, FEBackend const& fe_backend_, MatrixType const& mass_){
            using fe = typename Geometry::geo_map;
            using cub= typename Geometry::cub;
            using geo_map = typename Geometry::geo_map;

            auto d1=assembly_.m_d1;
            auto d2=assembly_.m_d2;
            auto d3=assembly_.m_d3;


            // [reference & comparison]
            FieldContainer<double> function_at_cub_points(fe::basisCardinality, cub::numCubPoints());

            for (uint_t i=0; i<fe::basisCardinality; ++i)
                for (uint_t j=0; j<cub::numCubPoints(); ++j)
                {
                	function_at_cub_points(i,j)=(*fe_backend_.m_phi_at_cub_points_s)(i,j,0);
                	std::cout<<function_at_cub_points(i,j)<<std::endl;
                }

            FieldContainer<double> cub_weights(cub::numCubPoints());

            for(uint_t i=0; i<cub::numCubPoints(); ++i)
            {
                cub_weights(i)=fe_backend_.m_cub_weights_s(i,0,0);
            }

            FieldContainer<double> cub_points(cub::numCubPoints(), fe::spaceDim);
            for(uint_t i=0; i<cub::numCubPoints(); ++i)
            {
                for(uint_t j=0; j<fe::spaceDim; ++j)
                {
                    cub_points(i,j)=fe_backend_.m_cub_points_s(i,j,0);
                }
            }

            FieldContainer<double> grid(d1*d2*d3, geo_map::basisCardinality, 3);

            for (int i =0; i< d1; ++i)
                for (int j =0; j< d2; ++j)
                    for (int k =0; k< d3; ++k)
                    {
                        for (int q=0; q<geo_map::basisCardinality; ++q)
                        {
                            for (int d=0; d<3; ++d)
                            {//assign
                                grid(i*d2*d3+j*d3+k, q, d)=assembly_base_.get_grid()(i, j, k, q, d);
                            }
                        }
                    }

            FieldContainer<double> jac((d1*d2*d3), cub::numCubPoints(), geo_map::spaceDim, geo_map::spaceDim);
            CellTools<double>::setJacobian(jac, cub_points, grid, geo_map::cell_t::value);

            auto epsilon=1e-15;

            std::cout<<" d1 d2 d3 numVub spaceDim "<<d1<<" "<<d2<<" "<<d3<<" "<<cub::numCubPoints()<<" "<<geo_map::spaceDim<<std::endl;

            for (int i =0; i< d1; ++i)
                for (int j =0; j< d2; ++j)
                    for (int k =0; k< d3; ++k)
                        for (int q=0; q<cub::numCubPoints(); ++q)
                        {
                            for (int dimx=0; dimx<geo_map::spaceDim; ++dimx)
                                for (int dimy=0; dimy<geo_map::spaceDim; ++dimy)
                                {

                                    std::cout<<"values in i="<<i<<" j="<<j<<" k="<<k<<" q="<<q<<" "<<dimx<<" "<<dimy<<": "
                                             <<assembly_.get_jac()(i, j, k, q, dimx, dimy)<<" and "
                                             <<jac(i*d3*d2+j*d3+k, q, dimx, dimy)// weighted_measure(i*d2*d3+j*d3+k, q)
                                             <<std::endl;

                                    if(assembly_.get_jac()(i, j, k, q, dimx, dimy) > epsilon+ jac(i*d2*d3+j*d3+k, q, dimx, dimy)/*weighted_measure(i*d2*d3+j*d3+k, q)*/
                                       ||
                                       assembly_.get_jac()(i, j, k, q, dimx, dimy) +epsilon < jac(i*d2*d3+j*d3+k, q, dimx, dimy)// weighted_measure(i*d2*d3+j*d3+k, q)
                                        ){
                                        std::cout<<"error in i="<<i<<" j="<<j<<" k="<<k<<" q="<<q<<" "<<dimx<<" "<<dimy<<": "
                                                 <<assembly_.get_jac()(i, j, k, q, dimx, dimy)<<" != "
                                                 <<jac(i*d3*d2+j*d3+k, q, dimx, dimy)// weighted_measure(i*d2*d3+j*d3+k, q)
                                                 <<std::endl;
                                        assert(false);

                                    }
                                }
                        }


            FieldContainer<double> jac_det((d1*d2*d3), cub::numCubPoints());

            // storage_t<layout_map<0,1,2> >::storage_t weighted_measure_s ((d1*d2*d3), cub::numCubPoints(), 1);
            FieldContainer<double> weighted_measure((d1*d2*d3), cub::numCubPoints());

            CellTools<double>::setJacobianDet(jac_det, jac);
            FunctionSpaceTools::computeCellMeasure<double>(weighted_measure,                      // compute weighted cell measure
                                                           jac_det,
                                                           cub_weights);


            for (int i =0; i< d1; ++i)
            {
                for (int j =0; j< d2; ++j)
                {
                    for (int k =0; k< d3; ++k)
                    {
                        for (int q=0; q<cub::numCubPoints(); ++q)
                        {

                            std::cout<<"Det val in i="<<i<<" j="<<j<<" k="<<k<<" q="<<q<<": "
                                     <<assembly_.get_jac_det()(i, j, k, q)<<" and "
                                     <<jac_det(i*d2*d3+j*d3+k, q)
                                     <<std::endl;


                            if(assembly_.get_jac_det()(i, j, k, q) > epsilon+ jac_det(i*d2*d3+j*d3+k, q)
                               ||
                               assembly_.get_jac_det()(i, j, k, q) +epsilon < jac_det(i*d2*d3+j*d3+k, q)
                                ){
                                std::cout<<"error in i="<<i<<" j="<<j<<" k="<<k<<" q="<<q<<": "
                                         <<assembly_.get_jac_det()(i, j, k, q)<<" != "
                                         <<jac_det(i*d2*d3+j*d3+k, q)
                                         <<std::endl;
                                assert(false);
                            }
                        }
                    }
                }
            }





            FieldContainer<double> weighted_function_at_cub_points((d1*d2*d3), fe::basisCardinality, cub::numCubPoints());
            weighted_function_at_cub_points.initialize(1.);

            FunctionSpaceTools::multiplyMeasure<double>(weighted_function_at_cub_points,  // multiply with weighted measure
                                                        weighted_measure,
														function_at_cub_points);



            FieldContainer<double> function_at_cub_points_ext((d1*d2*d3),fe::basisCardinality, cub::numCubPoints());
            for (int i =0; i< d1; ++i)
            {
                for (int j =0; j< d2; ++j)
                {
                    for (int k =0; k< d3; ++k)
                    {
                        for (int P_i =0; P_i< fe::basisCardinality; ++P_i)
                        {
                            for (int q=0; q<cub::numCubPoints(); ++q)
                            {
                            	function_at_cub_points_ext(i*d2*d3+j*d3+k, P_i, q) = function_at_cub_points(P_i,q);
                            }
                        }
                    }
                }
            }

            FieldContainer<double> mass_matrices((d1*d2*d3), fe::basisCardinality, fe::basisCardinality);
            FunctionSpaceTools::integrate<double>(mass_matrices,                             // compute mass matrices
            									  function_at_cub_points_ext,
												  weighted_function_at_cub_points,
                                                  Intrepid::COMP_CPP);

    //#ifndef REORDER
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

                                std::cout<<"values in i="<<i<<" j="<<j<<" k="<<k<<" P_i="<<P_i<<" Q_i="<<Q_i<<": "
                                         <<mass_(i, j, k, P_i, Q_i)<<" and "
                                         <<mass_matrices(i*d2*d3+j*d3+k, P_i, Q_i)
                                         <<std::endl;


                            	if(mass_(i, j, k, P_i, Q_i) > epsilon+ mass_matrices(i*d2*d3+j*d3+k, P_i, Q_i)
                                   ||
								   mass_(i, j, k, P_i, Q_i) +epsilon < mass_matrices(i*d2*d3+j*d3+k, P_i, Q_i)
                                    ){
                                    std::cout<<"error in i="<<i<<" j="<<j<<" k="<<k<<" P_i="<<P_i<<" Q_i="<<Q_i<<": "
                                             <<mass_(i, j, k, P_i, Q_i)<<" != "
                                             <<mass_matrices(i*d2*d3+j*d3+k, P_i, Q_i)
                                             <<std::endl;
                                    assert(false);
                                }
                            }
                    }
                }
            }
    //#endif


            // [reference & comparison]

            return true;
        }

}//namespace intrepid
//}//namespace gridtools
