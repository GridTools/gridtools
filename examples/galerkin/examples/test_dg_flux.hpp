/**
\file
*/
#pragma once
#include "../numerics/bd_assembly.hpp"
#include <Intrepid_CellTools.hpp>

namespace gridtools{
    namespace intrepid{
        using namespace Intrepid;

        template <typename Geometry, typename BdAssembly, typename BaseAssembly, typename FEBackend, typename MatrixType>
        bool test(BaseAssembly const& assembly_base_, BdAssembly const& assembly_, FEBackend const& fe_backend_, MatrixType const& flux_ ){
            using fe = typename Geometry::geo_map;
            using cub= typename Geometry::cub;
            using geo_map = typename Geometry::geo_map;
            using bd_geo_map = typename Geometry::bd_geo_map;

            auto d1=assembly_base_.m_d1;
            auto d2=assembly_base_.m_d2;
            auto d3=assembly_base_.m_d3;

            // [reference & comparison]
            FieldContainer<double> grad_at_cub_points(fe::basis_cardinality(), cub::numCubPoints(), fe::space_dim());

            for (uint_t i=0; i<fe::basis_cardinality(); ++i)
                for (uint_t j=0; j<cub::numCubPoints(); ++j)
                    for (int_t k=0; k<fe::space_dim(); ++k)
                        grad_at_cub_points(i,j,k)=fe_backend_.m_grad_at_cub_points(i,j,k,0);

            FieldContainer<double> cub_weights(cub::numCubPoints());

            for(uint_t i=0; i<cub::numCubPoints(); ++i)
            {
                cub_weights(i)=fe_backend_.m_rule.m_bd_cub_weights(i,0,0);
            }

            auto cub_points=fe_backend_.m_rule.update_boundary_cub(1);

            FieldContainer<double> grid(d1*d2*d3, geo_map::basis_cardinality(), 3);

            for (int i =0; i< d1; ++i)
                for (int j =0; j< d2; ++j)
                    for (int k =0; k< d3; ++k)
                    {
                        for (int q=0; q<geo_map::basis_cardinality(); ++q)
                        {
                            for (int d=0; d<3; ++d)
                            {//assign
                                grid(i*d2*d3+j*d3+k, q, d)=assembly_base_.get_grid()(i, j, k, q, d);
                            }
                        }
                    }

            FieldContainer<double> jac((d1*d2*d3), cub::numCubPoints(), geo_map::space_dim(), geo_map::space_dim());
            CellTools<double>::setJacobian(jac, cub_points, grid, geo_map::cell_t::value);

            auto epsilon=1e-15;

            std::cout<<"check boundary jacobian"<<"\n";

            for (int i =1; i< d1; ++i)
                for (int j =1; j< d2; ++j)
                    for (int k =0; k< d3; ++k)
                        for (int q=0; q<cub::numCubPoints(); ++q)
                        {
                            for (int dimx=0; dimx<geo_map::space_dim(); ++dimx)
                                for (int dimy=0; dimy<geo_map::space_dim(); ++dimy)
                                {
                                    if(assembly_.get_bd_jac()(i, j, k, q, dimx, dimy, 0) > epsilon+ jac(i*d2*d3+j*d3+k, q, dimx, dimy)/*weighted_measure(i*d2*d3+j*d3+k, q)*/
                                       ||
                                       assembly_.get_bd_jac()(i, j, k, q, dimx, dimy, 0) +epsilon < jac(i*d2*d3+j*d3+k, q, dimx, dimy)// weighted_measure(i*d2*d3+j*d3+k, q)
                                        )
                                    {
                                        std::cout<<"error in i="<<i<<" j="<<j<<" k="<<k<<" q="<<q<<" dimx="<<dimx<<" dimy="<<dimy<<": "
                                                 <<assembly_.get_bd_jac()(i, j, k, q, dimx, dimy, 0)<<" != "
                                                 <<jac(i*d3*d2+j*d3+k, q, dimx, dimy)// weighted_measure(i*d2*d3+j*d3+k, q)
                                                 <<"\n";
                                        // assert(false);
                                    }
                                }
                        }
            std::cout<<"done\n";



            // CellTopology hexahedron_8( shards::getCellTopologyData<shards::Hexahedron<8> >() );
            int worksetSize=d1*d2*d3; //number of cells
            int subcellOrd=1;
            //   Step 2.2.a: Allocate storage for face tangents and face normals
            FieldContainer<double> worksetFaceTu(worksetSize, cub::numCubPoints(), 3/*pCellDim*/);
            FieldContainer<double> worksetFaceTv(worksetSize, cub::numCubPoints(), 3);
            FieldContainer<double> worksetFaceN(worksetSize, cub::numCubPoints(), 3);

            //   Step 2.2.b: Compute face tangents
            CellTools<double>::getPhysicalFaceTangents(worksetFaceTu,
                                                       worksetFaceTv,
                                                       jac,
                                                       subcellOrd,
                                                       Geometry::geo_map::cell_t::value
                );

            //   Step 2.2.c: Face outer normals (relative to parent cell) are uTan x vTan:
            RealSpaceTools<double>::vecprod(worksetFaceN, worksetFaceTu, worksetFaceTv);
            std::cout<<"check face normals"<<"\n";
            for (int i =1; i< d1; ++i)
                for (int j =1; j< d2; ++j)
                    for (int k =0; k< d3; ++k)
                        for (int q=0; q<cub::numCubPoints(); ++q)
                        {
                            double norm1_squared=assembly_.get_normals()(i,j,k, q,0, 0)*assembly_.get_normals()(i,j,k, q,0, 0)
                                + assembly_.get_normals()(i,j,k, q,1, 0)*assembly_.get_normals()(i,j,k, q,1, 0)
                                + assembly_.get_normals()(i,j,k, q,2, 0)*assembly_.get_normals()(i,j,k, q,2, 0);
                            double norm2_squared=worksetFaceN(i*d2*d3+j*d3+k, q, 0)*worksetFaceN(i*d2*d3+j*d3+k, q, 0)
                                +worksetFaceN(i*d2*d3+j*d3+k, q, 1)*worksetFaceN(i*d2*d3+j*d3+k, q, 1)
                                +worksetFaceN(i*d2*d3+j*d3+k, q, 2)*worksetFaceN(i*d2*d3+j*d3+k, q, 2);

                            for (int dimx=0; dimx<geo_map::space_dim(); ++dimx)
                                if(worksetFaceN(i*d2*d3+j*d3+k, q, dimx)/sqrt(norm2_squared) > epsilon + assembly_.get_normals()(i,j,k, q,dimx, 0)/sqrt(norm1_squared)
                                   ||
                                   worksetFaceN(i*d2*d3+j*d3+k, q, dimx)/sqrt(norm2_squared) + epsilon < assembly_.get_normals()(i,j,k, q,dimx, 0)/sqrt(norm1_squared)
                                    )
                                    std::cout<<"error in i="<<i<<" j="<<j<<" k="<<k<<" q="<<q<<" "<<dimx<<": "
                                             <<assembly_.get_normals()(i, j, k, q, dimx, 0)/sqrt(norm1_squared)<<" != "
                                             <<worksetFaceN(i*d3*d2+j*d3+k, q, dimx)/sqrt(norm2_squared)// weighted_measure(i*d2*d3+j*d3+k, q)
                                             <<"\n";
                        }
            std::cout<<"done\n";


            FieldContainer<double> jac_det((d1*d2*d3), cub::numCubPoints());


        }
    }
}
