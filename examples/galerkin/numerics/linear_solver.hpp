#pragma once

#include "../galerkin_defs.hpp"
#include "../functors/matvec.hpp"
#include "../functors/reduction.hpp"
#include "basis_functions.hpp"
#include <memory>

namespace gdl {

    // TODO: implement CRTP pattern for linear solvers, this is just a placeholder for the time being
    /**
      @class Linear solver base struct
      @tparam Algorithmic strategy (Richardson, Conjugate Gradient, etc)
     */
    template <typename Strategy>
    struct linear_solver{

        /**
          @brief solve method implementation
          @tparam Matrix storage type
          @tparam Rhs vector storage type
          @tparam Unknowns vector storage type
          @param Matrix
          @param Rhs vector
          @param Unknowns vector (I/O parameter)
          @param Algorithm specific parameters
          */
        // TODO: some parameters should be const ref
        template <typename MatrixType_A, typename VectorType_b, typename VectorType_x, typename ... Params>
        static inline void solve(MatrixType_A & i_A,
                                 VectorType_b & i_b,
                                 VectorType_x & io_x,
                                 Params const & ... i_params)
        {
            // TODO: split loop structure and iteration operations
            Strategy::solve_impl(i_A, i_b, io_x, i_params...);
        }
    };

#define PRINT false

    /**
      @class Conjugate gradient solver functor

          This struct implements the conjugate gradient method for the solution
          of a linear system Ax=b.

          hypotheses:

          - the implemented procedure expects the A matrix and b vector to be
            unassembled, while the I/O unknows vector x must be provided and
            is returned in its assembled form.
            *** The x vector values corresponding to dofs shared by adjacent
            *** mesh elements must be correctly assigned to the corresponding
            *** storage elements (shared dof vector values must be duplicated)

          - the maximum number of iterations performed during solution search
            can be specified by the user.

          - the algorithm stops before the maximum number of iteration is reached
            if the solution error (square modulus of b-Ax for current x) or the solution
            stability (square modulus of x-x' for two subsequent iterations) are smaller
            than the provided threshold values

          // TODO: it would be nice to remove the following condition, at least from external interface
          - A, b and x storages must include a "right" and "left" halo filled by zero values

      @tparam Number of single mesh element dofs along x direction
      @tparam Number of single mesh element dofs along y direction
      @tparam Number of single mesh element dofs along z direction
      // TODO: extra parameter for mesh structure (hexahedrons, tetrahedrons, etc..) will be needed
      // TODO: add solver statistics printing in verbose mode (num iterations, error, stability, etc...)
     */
    template <ushort_t N_DOF0, ushort_t N_DOF1, ushort_t N_DOF2>
    struct cg_solver : public linear_solver< cg_solver<N_DOF0, N_DOF1, N_DOF2> > {


        /**
          @brief solve method implementation
          @tparam Matrix storage type
          @tparam Rhs vector storage type
          @tparam Unknowns vector storage type
          @param Unassembled Matrix
          @param Unassembled Rhs vector
          @param Assembled Unknowns vector (I/O parameter) (***see comment in class description)
          @param Solution stability threshold
          @param Solution error threshold
          @param Maximum number of iterations
         */
        template <typename MatrixType_A, typename VectorType_b, typename VectorType_x>
        static void solve_impl(MatrixType_A & i_A,// TODO: add constness
                               VectorType_b & i_b,// TODO: add constness
                               VectorType_x & io_x,
                               float_t const & i_stability_threshold,
                               float_t const & i_error_threshold,
                               uint_t const & i_max_iterations)
        {

            // Algorithm implementation strategy summary
            //
            //  - assembled version of A matrix and b vector is not required and assembling of other involved math
            //    object is avoided as much as possible. However in some cases it must be performed...
            //
            //  - A*v product: the vector v needs to be in its assembled version and the result is unassembled
            //
            //  - v_a*v_b scalar product: both vectors v_a and v_b need to be in assembled version
            //
            //  - Vector scalar product (including vector module calculation), is currently performed in 3 steps:
            //    1) calculation of element-wise multiplication (functors::vecvec<4,functors::mult_operator<float_t>)
            //    2) calculation of partial reduction which performs the sum of the single element-wise product
            //       belonging to a single mesh element/thread and stores the result in the 0-th element of the output vector
            //       (functors::partial_hexahedron_assembled_reduction<N_DOF0,N_DOF1,N_DOF2,functors::sum_operator<float_t> > )
            //
            //    3) final reduction with an external (non-GT based) sum over 0-th elements of the partial reduction
            //       output vector
            //
            //  - Vector assembling is currently performed in two steps:
            //    1) Sum of the contributions coming from different mesh elements/threads corresponding to the same
            //       dof and storing of the result in the storage elements corresponding to the mesh element/thread
            //       responsible for the considered dof (see assembly_functors.hpp info)
            //       (functors::hexahedron_vector_assemble<N_DOF0,N_DOF1,N_DOF2>)
            //    2) Redistribution of the assembled values to the mesh elements storages holding the same dofs but
            //       not responsible for them during assemble procedure. This last step is needed in particular when
            //       the assemble resulting vector is involved in a subsequent matrix*vector or vector*vector product



            // Computing grid definitions

            // Assemble related grid definition: this grid contains the "right" halo, for which the provided A matrix and b vector
            // are assumed to have values equal to zero (see struct comment). Calculation in the "right" halo are performed only to keep algorithm
            // uniformity wrt the computing thread.
            auto mesh_coords_ass=gridtools::grid<axis>({1, 0, 1, i_b.meta_data().template dim<0>()-1, i_b.meta_data().template dim<0>()},
                                                       {1, 0, 1, i_b.meta_data().template dim<1>()-1, i_b.meta_data().template dim<1>()});
            mesh_coords_ass.value_list[0] = 1;
            mesh_coords_ass.value_list[1] = i_b.meta_data().template dim<2>()-1;

            // Restricted grid definition: this grid does not contain any halo and is used for operations of non assemble-based type (see struct comment).
            auto mesh_coords=gridtools::grid<axis>({1, 0, 1, i_b.meta_data().template dim<0>()-2, i_b.meta_data().template dim<0>()-1},
                                                   {1, 0, 1, i_b.meta_data().template dim<1>()-2, i_b.meta_data().template dim<1>()-1});
            mesh_coords.value_list[0] = 1;
            mesh_coords.value_list[1] = i_b.meta_data().template dim<2>()-2;


            // Storage definition
            // Placeholders with "_ass_" sequence are expected to reference to quantities assembled on the considered
            // mesh. The input input/output unknowns vector is an example of this condition. On the other side if the
            // "_ass_" fragment is not present, the placeholder refers to an unassembled quantity, such as the A matrix
            // or the rhs vector b

            // TODO: reduce storage list
            // TODO: store scalars as scalars

            // Unassembled rhs vector for current x vector in Ax=b operation (This is unassembled because of the A matrix)
            VectorType_b b_test(i_b.meta_data(), 0., "b_test");

            // Assembled error vector r=b_test-b
            VectorType_b r_ass(i_b.meta_data(),0.,"r_ass");

            // Assembled error vector module (r*r before summing over the elements)
            VectorType_b r_mod_ass(i_b.meta_data(),0.,"r_mod_ass");

            // Assembled and partially reduced error vector module (r*r after summing over the elements of a single thread)
            VectorType_b r_mod_red_ass(i_b.meta_data(),0.,"r_mod_red_ass");

            // Assembled p vector (p'=r+beta*p)
            VectorType_b p_ass(i_b.meta_data(),0.,"p_ass");

            // Unassembled Ap product
            VectorType_b Ap(i_b.meta_data(),0.,"Ap");

            // Assembled Ap product
            VectorType_b Ap_ass(i_b.meta_data(),0.,"Ap_ass");

            // Assembled pAp product (before summing over elements)
            VectorType_b pAp_ass(i_b.meta_data(),0.,"pAp_ass");

            // Assembled and partially reduced error vector module (after summing over the elements of a single thread)
            VectorType_b pAp_red(i_b.meta_data(),0.,"pAp_red");

            // Alpha scalar factor alpha = r*r/pAp
            VectorType_b alpha(i_b.meta_data(),0.,"alpha");

            // Assembled alpha*p vector
            VectorType_b alphap_ass(i_b.meta_data(),0.,"alphap_ass");

            // Assembled alpha*p vector module ((alpha*p)*(alpha*p) before summing over the elements)
            VectorType_b alphap_mod_ass(i_b.meta_data(),0.,"alphap_mod_ass");

            // Assembled and partially reduced alpha*p vector module ((alpha*p)*(alpha*p) after summing over the elements of a single thread)
            VectorType_b alphap_mod_red_ass(i_b.meta_data(),0.,"alphap_mod_red_ass");

            // Assembled alpha*Ap vector
            VectorType_b alphaAp_ass(i_b.meta_data(),0.,"alphaAp_ass");

            // Assembled updated error vector module (r'*r' before summing over the elements)
            VectorType_b r_upd_mod_ass(i_b.meta_data(),0.,"r_upd_mod_ass");

            // Assembled and partially reduced updated error vector module (r'*r' after summing over the elements of a single thread)
            VectorType_b r_upd_mod_red_ass(i_b.meta_data(),0.,"r_upd_mod_red_ass");

            // Beta scalar factor beta = r'*r'/r*r
            VectorType_b beta(i_b.meta_data(),0.,"beta");

            // Assembled beta*p vector
            VectorType_b betap_ass(i_b.meta_data(),0.,"betap_ass");


            /////////////////////////////////
            // Compute r = b - Ax
            /////////////////////////////////

            typedef gt::arg<0,MatrixType_A> p_A;
            typedef gt::arg<1,VectorType_x> p_x_ass;
            typedef gt::arg<2,VectorType_b> p_b_test;
            typedef gt::arg<3,VectorType_b> p_i_b;
            typedef gt::arg<4,VectorType_b> p_b;
            typedef gt::arg<5,VectorType_b> p_r;
            typedef gt::arg<6,VectorType_b> p_r_ass;
            typedef gt::arg<7,VectorType_b> p_p_ass;
            typedef gt::arg<8,VectorType_b> p_r_mod_ass;
            typedef gt::arg<9,VectorType_b> p_r_mod_red_ass;
            typedef gt::arg<10,VectorType_b> p_Ap;
            typedef gt::arg<11,VectorType_b> p_Ap_ass;
            typedef gt::arg<12,VectorType_b> p_pAp_ass;
            typedef gt::arg<13,VectorType_b> p_pAp_red;
            typedef boost::mpl::vector<p_A, p_x_ass, p_b_test, p_i_b, p_b, p_r, p_r_ass, p_p_ass, p_r_mod_ass, p_r_mod_red_ass, p_Ap, p_Ap_ass, p_pAp_ass, p_pAp_red> start_accessor_list;
            gridtools::aggregator_type<start_accessor_list> domain(boost::fusion::make_vector(&i_A, &io_x, &b_test, &i_b, &i_b, &r_ass, &r_ass, &p_ass, &r_mod_ass, &r_mod_red_ass, &Ap, &Ap_ass, &pAp_ass, &pAp_red));

            auto compute_r=gridtools::make_computation<BACKEND>(domain,
                                                                mesh_coords,
                                                                gridtools::make_multistage(gridtools::enumtype::execute<gridtools::enumtype::forward>(),
                                                                                    gridtools::make_stage<functors::matvec>(p_A(),p_x_ass(),p_b_test()),
                                                                                    gridtools::make_stage<functors::vecvec<4,functors::sub_operator<float_t> > >(p_b(),p_b_test(),p_r())
                                                                ));


            compute_r->ready();
            compute_r->steady();
            compute_r->run();
            compute_r->finalize();
//	    compute_r->reset();

#if PRINT

            for(uint_t i=1;i<i_A.meta_data().template dim<0>()-1;++i)
                for(uint_t j=1;j<i_A.meta_data().template dim<1>()-1;++j)
                    for(uint_t k=1;k<i_A.meta_data().template dim<2>()-1;++k)
                        for(uint_t l=0;l<i_A.meta_data().template dim<3>();++l)
                            for(uint_t m=0;m<i_A.meta_data().template dim<4>();++m)
                                std::cout<<i<<" "<<j<<" "<<k<<" "<<l<<" "<<m<<" A "<<i_A(i,j,k,l,m)<<std::endl;
            std::cout<<std::endl;

            for(uint_t i=1;i<i_b.meta_data().template dim<0>()-1;++i)
                for(uint_t j=1;j<i_b.meta_data().template dim<1>()-1;++j)
                    for(uint_t k=1;k<i_b.meta_data().template dim<2>()-1;++k)
                        for(uint_t l=0;l<i_b.meta_data().template dim<3>();++l)
                            if(r_ass(i,j,k,l)!=0)
                                std::cout<<i<<" "<<j<<" "<<k<<" "<<l<<" r "<<r_ass(i,j,k,l)<<std::endl;
            std::cout<<std::endl;
#endif


            //////////////////////////////////
            // Compute r*r
            // Set p = r
            //////////////////////////////////
            // TODO: search for storage copy functor
            auto compute_r2=gridtools::make_computation<BACKEND>(domain,
                                                           mesh_coords_ass,
                                                           gridtools::make_multistage(gridtools::enumtype::execute<gridtools::enumtype::forward>(),
                                                                               gridtools::make_stage<functors::hexahedron_vector_assemble<N_DOF0,N_DOF1,N_DOF2> >(p_r(),p_r_ass()),
							                       gridtools::make_stage<functors::hexahedron_vector_distribute<N_DOF0,N_DOF1,N_DOF2> >(p_r_ass()),
                                                                               gridtools::make_stage<functors::copy_vector<N_DOF0*N_DOF1*N_DOF2> >(p_r_ass(),p_p_ass()), // TODO: avoid this copy
                                                                               gridtools::make_stage<functors::vecvec<4,functors::mult_operator<float_t> > >(p_p_ass(),p_r_ass(),p_r_mod_ass()),
                                                                               gridtools::make_stage<functors::partial_hexahedron_assembled_reduction<N_DOF0,N_DOF1,N_DOF2,functors::sum_operator<float_t> > >(p_r_mod_ass(),p_r_mod_red_ass())
                                                           ));
            compute_r2->ready();
            compute_r2->steady();
            compute_r2->run();
            compute_r2->finalize();
//	    compute_r2->reset();


            // TODO: temporary solution, use Carlos' reduction or implement cuda
            // kernel merging with partial_hexahedron_assembled_reduction functor.
            // Sum over 0-th element of each mesh element r_mod_red_ass vector,
            // where the r*r product contribution of a single mesh element has
            // been accumulated
            float_t r_scal_mod = 0.;
            for(uint_t i=1;i<i_b.meta_data().template dim<0>();++i)
                for(uint_t j=1;j<i_b.meta_data().template dim<1>();++j)
                    for(uint_t k=1;k<i_b.meta_data().template dim<2>();++k)
                        r_scal_mod += r_mod_red_ass(i,j,k,0);


#if PRINT
            for(uint_t i=1;i<i_b.meta_data().template dim<0>();++i)
                for(uint_t j=1;j<i_b.meta_data().template dim<1>();++j)
                    for(uint_t k=1;k<i_b.meta_data().template dim<2>();++k)
                        for(uint_t l=0;l<i_b.meta_data().template dim<3>();++l)
                            if(r_ass(i,j,k,l)!=0)
                                std::cout<<i<<" "<<j<<" "<<k<<" "<<l<<" r_ass "<<r_ass(i,j,k,l)<<std::endl;
            std::cout<<std::endl;

            for(uint_t i=1;i<i_b.meta_data().template dim<0>();++i)
                for(uint_t j=1;j<i_b.meta_data().template dim<1>();++j)
                    for(uint_t k=1;k<i_b.meta_data().template dim<2>();++k)
                        for(uint_t l=0;l<i_b.meta_data().template dim<3>();++l)
                            if(r_mod_ass(i,j,k,l)!=0)
                                std::cout<<i<<" "<<j<<" "<<k<<" "<<l<<" r_mod_ass "<<r_mod_ass(i,j,k,l)<<std::endl;
            std::cout<<std::endl;

            for(uint_t i=1;i<i_b.meta_data().template dim<0>();++i)
                for(uint_t j=1;j<i_b.meta_data().template dim<1>();++j)
                    for(uint_t k=1;k<i_b.meta_data().template dim<2>();++k)
                        for(uint_t l=0;l<i_b.meta_data().template dim<3>();++l)
                            if(r_mod_red_ass(i,j,k,l)!=0)
                                std::cout<<i<<" "<<j<<" "<<k<<" "<<l<<" r_mod_red_ass "<<r_mod_red_ass(i,j,k,l)<<std::endl;
            std::cout<<std::endl;

            std::cout<<" r_scal_mod "<<r_scal_mod<<std::endl;
            std::cout<<std::endl;
#endif

            ////////////////////////////////////////////
            // Check if error is already below threshold
            ////////////////////////////////////////////
            if(r_scal_mod<i_error_threshold)
                return;


            uint_t iteration(0);
            float_t stability(0.);
            do{

                // TODO: error, stability, iteration, etc logging required

                iteration++;

                //////////////////////////////////
                // Compute Ap
                //////////////////////////////////

                auto compute_pAp=gridtools::make_computation<BACKEND>(domain,
                                                                      mesh_coords,
                                                                      gridtools::make_multistage(gridtools::enumtype::execute<gridtools::enumtype::forward>(),
                                                                                          gridtools::make_stage<functors::matvec>(p_A(),p_p_ass(),p_Ap())
                                                                      ));
                compute_pAp->ready();
                compute_pAp->steady();
                compute_pAp->run();
                compute_pAp->finalize();
//		compute_pAp->reset();

#if PRINT
                for(uint_t i=1;i<i_b.meta_data().template dim<0>()-1;++i)
                    for(uint_t j=1;j<i_b.meta_data().template dim<1>()-1;++j)
                        for(uint_t k=1;k<i_b.meta_data().template dim<2>()-1;++k)
                            for(uint_t l=0;l<i_b.meta_data().template dim<3>();++l)
                                if(p_ass(i,j,k,l)!=0)
                                    std::cout<<i<<" "<<j<<" "<<k<<" "<<l<<" p_p_ass "<<p_ass(i,j,k,l)<<std::endl;
                std::cout<<std::endl;


                for(uint_t i=1;i<i_b.meta_data().template dim<0>()-1;++i)
                    for(uint_t j=1;j<i_b.meta_data().template dim<1>()-1;++j)
                        for(uint_t k=1;k<i_b.meta_data().template dim<2>()-1;++k)
                            for(uint_t l=0;l<i_b.meta_data().template dim<3>();++l)
                                if(Ap(i,j,k,l)!=0)
                                    std::cout<<i<<" "<<j<<" "<<k<<" "<<l<<" Ap "<<Ap(i,j,k,l)<<std::endl;
                std::cout<<std::endl;

                for(uint_t i=1;i<i_b.meta_data().template dim<0>()-1;++i)
                    for(uint_t j=1;j<i_b.meta_data().template dim<1>()-1;++j)
                        for(uint_t k=1;k<i_b.meta_data().template dim<2>()-1;++k)
                            for(uint_t l=0;l<i_b.meta_data().template dim<3>();++l)
                                if(Ap_ass(i,j,k,l)!=0)
                                    std::cout<<i<<" "<<j<<" "<<k<<" "<<l<<" Ap_ass "<<Ap_ass(i,j,k,l)<<std::endl;
                std::cout<<std::endl;

#endif


                // TODO: these two steps for calculation of pAp are split because I am using two different grids,
                // in order to avoid useless calculation. What is the best practice in these cases?
                //////////////////////////////////
                // Compute pAp
                //////////////////////////////////

                // TODO: in principle p_Ap() to p_Ap_ass() assembly using 2 storages should be avoidable but I think that a
                // synchronization check must be introduced in that case
                auto compute_pAp2=gridtools::make_computation<BACKEND>(domain,
                                                                 mesh_coords_ass,
                                                                 gridtools::make_multistage(gridtools::enumtype::execute<gridtools::enumtype::forward>(),
                                                                                     gridtools::make_stage<functors::copy_vector<N_DOF0*N_DOF1*N_DOF2> >(p_Ap(),p_Ap_ass()),
                                                                                     gridtools::make_stage<functors::hexahedron_vector_assemble<N_DOF0,N_DOF1,N_DOF2> >(p_Ap(),p_Ap_ass()),
                                                                                     gridtools::make_stage<functors::hexahedron_vector_distribute<N_DOF0,N_DOF1,N_DOF2> >(p_Ap_ass()),
                                                                                     gridtools::make_stage< functors::assign< 4,zero<float_t> > >(p_Ap()),
                                                                                     gridtools::make_stage<functors::vecvec<4,functors::mult_operator<float_t> > >(p_p_ass(),p_Ap_ass(),p_pAp_ass()),
                                                                                     gridtools::make_stage<functors::partial_hexahedron_assembled_reduction<N_DOF0,N_DOF1,N_DOF2,functors::sum_operator<float_t> > >(p_pAp_ass(),p_pAp_red())
                                                                ));
                compute_pAp2->ready();
                compute_pAp2->steady();
                compute_pAp2->run();
                compute_pAp2->finalize();
//		compute_pAp2->reset();

                // TODO: temporary solution, use Carlos' reduction or implement cuda
                // kernel merging with partial_hexahedron_assembled_reduction functor.
                // Sum over 0-th element of each mesh element pAp vector,
                // where the pAp product contribution of a single mesh element has
                // been accumulated
                float_t pAp_scal = 0.;
                for(uint_t i=1;i<i_b.meta_data().template dim<0>();++i)
                    for(uint_t j=1;j<i_b.meta_data().template dim<1>();++j)
                        for(uint_t k=1;k<i_b.meta_data().template dim<2>();++k)
                            pAp_scal += pAp_red(i,j,k,0);

#if PRINT
                for(uint_t i=1;i<i_b.meta_data().template dim<0>();++i)
                    for(uint_t j=1;j<i_b.meta_data().template dim<1>();++j)
                        for(uint_t k=1;k<i_b.meta_data().template dim<2>();++k)
                            for(uint_t l=0;l<i_b.meta_data().template dim<3>();++l)
                                if(Ap_ass(i,j,k,l)!=0)
                                    std::cout<<i<<" "<<j<<" "<<k<<" "<<l<<" Ap_ass "<<Ap_ass(i,j,k,l)<<std::endl;
                std::cout<<std::endl;


                for(uint_t i=1;i<i_b.meta_data().template dim<0>();++i)
                    for(uint_t j=1;j<i_b.meta_data().template dim<1>();++j)
                        for(uint_t k=1;k<i_b.meta_data().template dim<2>();++k)
                            for(uint_t l=0;l<i_b.meta_data().template dim<3>();++l)
                                if(pAp_ass(i,j,k,l)!=0)
                                    std::cout<<i<<" "<<j<<" "<<k<<" "<<l<<" pAp_ass "<<pAp_ass(i,j,k,l)<<std::endl;
                std::cout<<std::endl;


                for(uint_t i=1;i<i_b.meta_data().template dim<0>();++i)
                    for(uint_t j=1;j<i_b.meta_data().template dim<1>();++j)
                        for(uint_t k=1;k<i_b.meta_data().template dim<2>();++k)
                            for(uint_t l=0;l<i_b.meta_data().template dim<3>();++l)
                                if(pAp_red(i,j,k,l)!=0)
                                    std::cout<<i<<" "<<j<<" "<<k<<" "<<l<<" pAp_red "<<pAp_red(i,j,k,l)<<std::endl;
                std::cout<<std::endl;

                std::cout<<" pAp_scal "<<pAp_scal<<std::endl;
                std::cout<<std::endl;
#endif


                //////////////////////////////////
                // Compute alpha = r*r/pAp
                //////////////////////////////////
                const float_t alpha_scal(r_scal_mod/pAp_scal);

                // TODO: temporary solution, use global_accessor
                for(uint_t i=1;i<i_b.meta_data().template dim<0>();++i)
                    for(uint_t j=1;j<i_b.meta_data().template dim<1>();++j)
                        for(uint_t k=1;k<i_b.meta_data().template dim<2>();++k)
                            for(uint_t l=0;l<i_b.meta_data().template dim<3>();++l)
                                alpha(i,j,k,l) = alpha_scal;

#if PRINT
                std::cout<<" alpha_scal "<<alpha_scal<<std::endl;
                std::cout<<std::endl;

                for(uint_t i=1;i<i_b.meta_data().template dim<0>();++i)
                    for(uint_t j=1;j<i_b.meta_data().template dim<1>();++j)
                        for(uint_t k=1;k<i_b.meta_data().template dim<2>();++k)
                            for(uint_t l=0;l<i_b.meta_data().template dim<3>();++l)
                                if(r_ass(i,j,k,l)!=0)
                                    std::cout<<i<<" "<<j<<" "<<k<<" "<<l<<" p_r_ass_xupd "<<r_ass(i,j,k,l)<<std::endl;
                std::cout<<std::endl;
#endif


                /////////////////////////////////////
                // Compute x' = x + alpha*p
                // Compute r' = r - alpha*Ap*
                // Compute r'*r'
                // Compute (alpha*p)*(alpha*p) = dx^2
                // Compute beta = r'*r'/r*r
                /////////////////////////////////////

                typedef gt::arg<0, VectorType_b> p_p_ass_xupd;
                typedef gt::arg<1, VectorType_b> p_alpha_xupd;
                typedef gt::arg<2, VectorType_b> p_alphap_ass_xupd;
                typedef gt::arg<3, VectorType_b> p_alphap_ass_copy_xupd;
                typedef gt::arg<4, VectorType_x> p_x_ass_xupd;
                typedef gt::arg<5, VectorType_x> p_x_upd_ass_xupd;
                typedef gt::arg<6, VectorType_b> p_Ap_ass_xupd;
                typedef gt::arg<7, VectorType_b> p_alphaAp_ass_xupd;
                typedef gt::arg<8, VectorType_b> p_r_ass_xupd;
                typedef gt::arg<9, VectorType_b> p_r_upd_ass_xupd;
                typedef gt::arg<10, VectorType_b> p_r_upd_ass_copy_xupd;
                typedef gt::arg<11, VectorType_b> p_r_upd_mod_ass_xupd;
                typedef gt::arg<12, VectorType_b> p_r_upd_mod_red_ass_xupd;
                typedef gt::arg<13, VectorType_b> p_alphap_mod_ass_xupd;
                typedef gt::arg<14, VectorType_b> p_alphap_mod_red_ass_xupd;
                typedef boost::mpl::vector<p_p_ass_xupd, p_alpha_xupd, p_alphap_ass_xupd, p_alphap_ass_copy_xupd, p_x_ass_xupd, p_x_upd_ass_xupd, p_Ap_ass_xupd, p_alphaAp_ass_xupd, p_r_ass_xupd, p_r_upd_ass_xupd, p_r_upd_ass_copy_xupd, p_r_upd_mod_ass_xupd, p_r_upd_mod_red_ass_xupd, p_alphap_mod_ass_xupd, p_alphap_mod_red_ass_xupd> x_update_accessor_list;
                gridtools::aggregator_type<x_update_accessor_list> x_update_domain(boost::fusion::make_vector(&p_ass, &alpha, &alphap_ass, &alphap_ass, &io_x, &io_x, &Ap_ass, &alphaAp_ass, &r_ass, &r_ass, &r_ass, &r_upd_mod_ass, &r_upd_mod_red_ass, &alphap_mod_ass, &alphap_mod_red_ass));



                auto compute_x_update=gridtools::make_computation<BACKEND>(x_update_domain,
                                                                           mesh_coords_ass,
                                                                           gridtools::make_multistage(gridtools::enumtype::execute<gridtools::enumtype::forward>(),
                                                                                               gridtools::make_stage<functors::vecvec<4,functors::mult_operator<float_t> > >(p_p_ass_xupd(),p_alpha_xupd(),p_alphap_ass_xupd()),
                                                                                               gridtools::make_stage<functors::vecvec<4,functors::sum_operator<float_t> > >(p_x_ass_xupd(),p_alphap_ass_xupd(),p_x_upd_ass_xupd()),
                                                                                               gridtools::make_stage<functors::vecvec<4,functors::mult_operator<float_t> > >(p_Ap_ass_xupd(),p_alpha_xupd(),p_alphaAp_ass_xupd()),
                                                                                               gridtools::make_stage<functors::vecvec<4,functors::sub_operator<float_t> > >(p_r_ass_xupd(),p_alphaAp_ass_xupd(),p_r_upd_ass_xupd()),
                                                                                               gridtools::make_stage<functors::vecvec<4,functors::mult_operator<float_t> > >(p_r_upd_ass_xupd(),p_r_upd_ass_copy_xupd(),p_r_upd_mod_ass_xupd()),
                                                                                               gridtools::make_stage<functors::vecvec<4,functors::mult_operator<float_t> > >(p_alphap_ass_xupd(),p_alphap_ass_copy_xupd(),p_alphap_mod_ass_xupd()),
                                                                                               gridtools::make_stage<functors::partial_hexahedron_assembled_reduction<N_DOF0,N_DOF1,N_DOF2,functors::sum_operator<float_t> > >(p_r_upd_mod_ass_xupd(),p_r_upd_mod_red_ass_xupd()),
                                                                                               gridtools::make_stage<functors::partial_hexahedron_assembled_reduction<N_DOF0,N_DOF1,N_DOF2,functors::sum_operator<float_t> > >(p_alphap_mod_ass_xupd(),p_alphap_mod_red_ass_xupd())
                                                                           ));

                compute_x_update->ready();
                compute_x_update->steady();
                compute_x_update->run();
                compute_x_update->finalize();


                // TODO: temporary solution, use Carlos' reduction or implement cuda
                // kernel merging with partial_hexahedron_assembled_reduction functor
                // Sum over 0-th element of each mesh element r_upd_mod_red_ass vector,
                // where the r'*r' product contribution of a single mesh element has
                // been accmulated
                float_t r_upd_scal_mod = 0.;
                for(uint_t i=1;i<i_b.meta_data().template dim<0>();++i)
                    for(uint_t j=1;j<i_b.meta_data().template dim<1>();++j)
                        for(uint_t k=1;k<i_b.meta_data().template dim<2>();++k)
                            r_upd_scal_mod += r_upd_mod_red_ass(i,j,k,0);

                // Sum over 0-th element of each mesh element alphap_mod_red_ass vector,
                // where the (alpha*p)*(alpha*p) product contribution of a single mesh element has
                // been accmulated
                stability = 0.;
                for(uint_t i=1;i<i_b.meta_data().template dim<0>();++i)
                    for(uint_t j=1;j<i_b.meta_data().template dim<1>();++j)
                        for(uint_t k=1;k<i_b.meta_data().template dim<2>();++k)
                            stability += alphap_mod_red_ass(i,j,k,0);
#if PRINT
                std::cout<<"Stability "<<stability<<std::endl;
                std::cout<<"Error "<<r_upd_scal_mod<<std::endl;
#endif
                if(r_upd_scal_mod<i_error_threshold)
                    break;

                const float_t beta_scal(r_upd_scal_mod/r_scal_mod);
                r_scal_mod = r_upd_scal_mod;

                // TODO: temporary solution, use global_accessor
                for(uint_t i=1;i<i_b.meta_data().template dim<0>();++i)
                    for(uint_t j=1;j<i_b.meta_data().template dim<1>();++j)
                        for(uint_t k=1;k<i_b.meta_data().template dim<2>();++k)
                            for(uint_t l=0;l<i_b.meta_data().template dim<3>();++l)
                                beta(i,j,k,l) = beta_scal;


#if PRINT
                for(uint_t i=1;i<i_b.meta_data().template dim<0>();++i)
                    for(uint_t j=1;j<i_b.meta_data().template dim<1>();++j)
                        for(uint_t k=1;k<i_b.meta_data().template dim<2>();++k)
                            for(uint_t l=0;l<i_b.meta_data().template dim<3>();++l)
                                if(io_x(i,j,k,l)!=0)
                                    std::cout<<i<<" "<<j<<" "<<k<<" "<<l<<" p_x_upd_ass_xupd "<<io_x(i,j,k,l)<<std::endl;
                std::cout<<std::endl;


                for(uint_t i=1;i<i_b.meta_data().template dim<0>();++i)
                    for(uint_t j=1;j<i_b.meta_data().template dim<1>();++j)
                        for(uint_t k=1;k<i_b.meta_data().template dim<2>();++k)
                            for(uint_t l=0;l<i_b.meta_data().template dim<3>();++l)
                                if(alphaAp_ass(i,j,k,l)!=0)
                                    std::cout<<i<<" "<<j<<" "<<k<<" "<<l<<" p_alphaAp_ass_xupd "<<alphaAp_ass(i,j,k,l)<<std::endl;
                std::cout<<std::endl;


                for(uint_t i=1;i<i_b.meta_data().template dim<0>();++i)
                    for(uint_t j=1;j<i_b.meta_data().template dim<1>();++j)
                        for(uint_t k=1;k<i_b.meta_data().template dim<2>();++k)
                            for(uint_t l=0;l<i_b.meta_data().template dim<3>();++l)
                                if(r_ass(i,j,k,l)!=0)
                                    std::cout<<i<<" "<<j<<" "<<k<<" "<<l<<" p_r_upd_ass_xupd "<<r_ass(i,j,k,l)<<std::endl;
                std::cout<<std::endl;


                for(uint_t i=1;i<i_b.meta_data().template dim<0>();++i)
                    for(uint_t j=1;j<i_b.meta_data().template dim<1>();++j)
                        for(uint_t k=1;k<i_b.meta_data().template dim<2>();++k)
                            for(uint_t l=0;l<i_b.meta_data().template dim<3>();++l)
                                if(r_upd_mod_ass(i,j,k,l)!=0)
                                    std::cout<<i<<" "<<j<<" "<<k<<" "<<l<<" p_r_upd_mod_ass_xupd "<<r_upd_mod_ass(i,j,k,l)<<std::endl;
                std::cout<<std::endl;

                for(uint_t i=1;i<i_b.meta_data().template dim<0>();++i)
                    for(uint_t j=1;j<i_b.meta_data().template dim<1>();++j)
                        for(uint_t k=1;k<i_b.meta_data().template dim<2>();++k)
                            for(uint_t l=0;l<i_b.meta_data().template dim<3>();++l)
                                if(alphap_mod_ass(i,j,k,l)!=0)
                                    std::cout<<i<<" "<<j<<" "<<k<<" "<<l<<" p_alphap_mod_ass_xupd "<<alphap_mod_ass(i,j,k,l)<<std::endl;
                std::cout<<std::endl;

                for(uint_t i=1;i<i_b.meta_data().template dim<0>();++i)
                    for(uint_t j=1;j<i_b.meta_data().template dim<1>();++j)
                        for(uint_t k=1;k<i_b.meta_data().template dim<2>();++k)
                            for(uint_t l=0;l<i_b.meta_data().template dim<3>();++l)
                                if(r_upd_mod_red_ass(i,j,k,l)!=0)
                                    std::cout<<i<<" "<<j<<" "<<k<<" "<<l<<" p_r_upd_mod_red_ass_xupd "<<r_upd_mod_red_ass(i,j,k,l)<<std::endl;
                std::cout<<std::endl;

                for(uint_t i=1;i<i_b.meta_data().template dim<0>();++i)
                    for(uint_t j=1;j<i_b.meta_data().template dim<1>();++j)
                        for(uint_t k=1;k<i_b.meta_data().template dim<2>();++k)
                            for(uint_t l=0;l<i_b.meta_data().template dim<3>();++l)
                                if(alphap_mod_red_ass(i,j,k,l)!=0)
                                    std::cout<<i<<" "<<j<<" "<<k<<" "<<l<<" p_alphap_mod_red_ass_xupd "<<alphap_mod_red_ass(i,j,k,l)<<std::endl;
                std::cout<<std::endl;

                std::cout<<" r_upd_scal_mod "<<r_upd_scal_mod<<std::endl;
                std::cout<<std::endl;

                std::cout<<" beta_scal "<<beta_scal<<std::endl;
                std::cout<<std::endl;
#endif

                /////////////////////////////////////
                // Compute p' = r + beta*p
                /////////////////////////////////////

                typedef gt::arg<0, VectorType_b> p_p_ass_pupd;
                typedef gt::arg<1, VectorType_b> p_beta_pupd;
                typedef gt::arg<2, VectorType_b> p_betap_ass_pupd;
                typedef gt::arg<3, VectorType_b> p_r_upd_ass_pupd;
                typedef boost::mpl::vector<p_p_ass_pupd, p_beta_pupd, p_betap_ass_pupd, p_r_upd_ass_pupd> p_update_accessor_list;
                gridtools::aggregator_type<p_update_accessor_list> p_update_domain(boost::fusion::make_vector(&p_ass, &beta, &betap_ass, &r_ass));


                auto compute_p_update=gridtools::make_computation<BACKEND>(p_update_domain,
                                                                           mesh_coords_ass,
                                                                           gridtools::make_multistage(gridtools::enumtype::execute<gridtools::enumtype::forward>(),
                                                                                               gridtools::make_stage<functors::vecvec<4,functors::mult_operator<float_t> > >(p_p_ass_pupd(),p_beta_pupd(),p_betap_ass_pupd()),
                                                                                               gridtools::make_stage<functors::vecvec<4,functors::sum_operator<float_t> > >(p_r_upd_ass_pupd(),p_betap_ass_pupd(),p_p_ass_pupd())
                                                                           ));
                compute_p_update->ready();
                compute_p_update->steady();
                compute_p_update->run();
                compute_p_update->finalize();

#if PRINT
                for(uint_t i=1;i<i_b.meta_data().template dim<0>();++i)
                    for(uint_t j=1;j<i_b.meta_data().template dim<1>();++j)
                        for(uint_t k=1;k<i_b.meta_data().template dim<2>();++k)
                            for(uint_t l=0;l<i_b.meta_data().template dim<3>();++l)
                                if(p_ass(i,j,k,l)!=0)
                                    std::cout<<i<<" "<<j<<" "<<k<<" "<<l<<" p_p_ass_pupd "<<p_ass(i,j,k,l)<<std::endl;
                std::cout<<std::endl;
#endif

            }while(stability>i_stability_threshold && iteration<i_max_iterations);

        }
    };


#if 0
    template <typename IterationStrategy>
    struct iteration_parameters;

    template <typename IterationStrategy>
    struct iterative_linear_solver : public linear_solver< iterative_linear_solver<IterationStrategy> > {

        template <typename MatrixType_A, typename VectorType_b, typename VectorType_x>
        static void solve_impl(MatrixType_A & i_A,
                               VectorType_b & i_b,
                               VectorType_x & io_x,
                               float_t const & i_stability_threshold,
                               uint_t const & i_max_iterations)
        {

            VectorType_b x_upd(i_b.meta_data(),0.,"x_upd");

            iteration_parameters<IterationStrategy> iter_pars;
            IterationStrategy::set_parameters(i_A,i_b,iter_pars);

            uint_t iteration(0);
            float_t stability(0.e0);

            do{
                iteration++;

                IterationStrategy::iterate(i_A,i_b,io_x,iter_pars,x_upd,stability);

                // TODO: "standard" this is wrong because of duplicated elements!
                // TODO: find best way to implement reduction: assembled/non assembled, etc... We could implement both using contained traits to decide which to use

            }while(stability>i_stability_threshold && iteration<i_max_iterations);

        }

    };

    template <ushort_t N_DOF0, ushort_t N_DOF1, ushort_t N_DOF2>
    struct richardson_iteration {

        template <typename MatrixType_A, typename VectorType_b>
        static inline void set_parameters(MatrixType_A & i_A,
                                          VectorType_b & i_b,
                                          iteration_parameters<richardson_iteration> & io_params)
        {
            io_params.m_omega = 0.1;
        }


        template <typename MatrixType_A, typename VectorType_b, typename VectorType_x>
        static inline void iterate(MatrixType_A & i_A,
                                   VectorType_b & i_b,
                                   VectorType_x & i_x,
                                   iteration_parameters<richardson_iteration> const & i_params,
                                   VectorType_b & io_x_upd,
                                   float_t & io_stability)
        {

            VectorType_b b_test(i_b.meta_data(), 0.e0, "b_test");
            VectorType_b err(i_b.meta_data(), 0.e0, "err");
            VectorType_b scaled_err(i_b.meta_data(), 0.e0, "scaled_err");
            VectorType_b scaled_err_ass(i_b.meta_data(), 0.e0, "scaled_err_ass");
            VectorType_b scaled_err_ass_copy(i_b.meta_data(), 0.e0, "scaled_err_ass_copy");
            VectorType_b omega(i_b.meta_data(), i_params.m_omega, "omega");

            auto mesh_coords=gridtools::grid<axis>({1, 0, 1, i_b.meta_data().template dim<0>()-1, i_b.meta_data().template dim<0>()},
                                                   {1, 0, 1, i_b.meta_data().template dim<1>()-1, i_b.meta_data().template dim<1>()});
            mesh_coords.value_list[0] = 1;
            mesh_coords.value_list[1] = i_b.meta_data().template dim<2>()-1;

            // TODO: reduce number of containers and copies
            typedef gt::arg<0, MatrixType_A> p_A;
            typedef gt::arg<1, VectorType_b> p_b;
            typedef gt::arg<2, VectorType_b> p_b_test;
            typedef gt::arg<3, VectorType_x> p_x;
            typedef gt::arg<4, VectorType_b> p_x_upd;
            typedef gt::arg<5, VectorType_b> p_err;
            typedef gt::arg<6, VectorType_b> p_scaled_err;
            typedef gt::arg<7, VectorType_b> p_scaled_err_ass;
            typedef gt::arg<8, VectorType_b> p_scaled_err_ass_copy;
            typedef gt::arg<9, VectorType_b> p_omega;
            typedef boost::mpl::vector<p_A, p_b, p_b_test, p_x, p_x_upd, p_err, p_scaled_err, p_scaled_err_ass, p_scaled_err_ass_copy, p_omega> accessor_list;
            gridtools::domain_type<accessor_list> domain(boost::fusion::make_vector(&i_A, &i_b, &b_test, &i_x, &io_x_upd, &err, &scaled_err, &scaled_err_ass, &scaled_err_ass, &omega));


            gt::dimension<1>::Index I;
            gt::dimension<2>::Index J;
            gt::dimension<3>::Index K;
            gt::dimension<4>::Index R;

            auto richardson_iteration=gridtools::make_computation<BACKEND>(domain,
                                                                           mesh_coords,
                                                                           gridtools::make_multistage(gridtools::enumtype::execute<gridtools::enumtype::forward>(),
                                                                                               gridtools::make_stage<functors::matvec>(p_A(),p_x(),p_b_test()),
                                                                                               gridtools::make_stage<functors::vecvec<4,functors::sub_operator<float_t> > >(p_b(),p_b_test(),p_err()),
											       gridtools::make_stage<functors::vecvec<4,functors::mult_operator<float_t> > >(p_omega(),p_err(),p_scaled_err()),
                                                                                               gridtools::make_stage<functors::tmp_copy_vector<N_DOF0*N_DOF1*N_DOF2> >(p_scaled_err(),p_scaled_err_ass()),
                                                                                               gridtools::make_stage<functors::hexahedron_vector_assemble<N_DOF0,N_DOF1,N_DOF2> >(p_scaled_err(),p_scaled_err_ass()),
                                                                                               gridtools::make_stage<functors::vecvec<4, functors::sum_operator<float_t> > >(p_x(),p_scaled_err_ass(),p_x_upd()),
                                                                                               gridtools::make_stage<functors::tmp_copy_vector<N_DOF0*N_DOF1*N_DOF2> >(p_x_upd(),p_x()),// TODO: avoid useless copy
                                                                                               gridtools::make_stage<functors::tmp_copy_vector<N_DOF0*N_DOF1*N_DOF2> >(p_scaled_err_ass(),p_scaled_err_ass_copy()),// TODO: avoid useless copy
                                                                                               gridtools::make_stage<functors::vecvec<4, functors::mult_operator<float_t> > >(p_scaled_err_ass(),p_scaled_err_ass_copy(),p_scaled_err()),
                                                                                               gridtools::make_stage<functors::partial_hexahedron_assembled_reduction<N_DOF0,N_DOF1,N_DOF2,functors::sum_operator<float_t> > >(p_scaled_err(),p_scaled_err_ass())
                                                                           ));
            richardson_iteration->ready();
            richardson_iteration->steady();
            richardson_iteration->run();
            richardson_iteration->finalize();

            // TODO: temporary solution
            io_stability = 0.;
            for(uint_t i=0;i<i_b.meta_data().template dim<0>();++i)
                for(uint_t j=0;j<i_b.meta_data().template dim<1>();++j)
                    for(uint_t k=0;k<i_b.meta_data().template dim<2>();++k)
                        io_stability += scaled_err_ass(i,j,k,0);

        }
    };

    template<>
    template < ushort_t ... Sizes >
    struct iteration_parameters<richardson_iteration<Sizes ... > > {
        float_t m_omega;
    };
#endif

#if 0

        template <typename Strategy>
        struct linear_solver{

            template <typename ... Params>
            static inline void solve(uint_t const i_size, float_t const * i_A, float_t const * i_b, float_t * io_x, Params const & ... i_params){ Strategy::solve_impl(i_size, i_A, i_b, io_x, i_params...); }

        };



        template <typename IterationStrategy>
        struct iterative_linear_solver : public linear_solver<iterative_linear_solver<IterationStrategy> > {

            static void solve_impl(uint_t const i_size, float_t const * i_A, float_t const * i_b, float_t * io_x, float_t const & i_stability_threshold, uint_t const & i_max_iterations){

                uint_t iteration(0);
                std::unique_ptr<float_t[]> x_upd(new float_t[i_size]);

                float_t stability(0.);
                do{
                    iteration++;

                    float_t omega(1.);
                    IterationStrategy::iterate(i_size,i_A,i_b,io_x,omega,x_upd.get());

                    //![check_stability]
                    stability = 0.;
                    for(uint_t i=0; i<i_size; i++)
                        stability += (io_x[i] - x_upd[i])*(io_x[i] - x_upd[i]);
                    //![check_stability]

                    //![assemble_update]
                    for(uint_t i=0; i<i_size; i++)
                       io_x[i] = x_upd[i];
                    //![assemble_and_update]hexahedron_vector_assemble

                }while(stability>i_stability_threshold && iteration<i_max_iterations);
            }
        };

        struct richardson_iteration {

            static inline void iterate(uint_t const i_size, float_t const * i_A, float_t const * i_b, float_t const * i_x, float_t i_omega, float_t * io_x_upd){

                for(uint_t i=0; i<i_size; i++)
                    for(uint_t j=0; j<i_size; j++)
                        io_x_upd[i] = i_x[i] + i_omega*(i_b[i] - i_A[i*i_size+j]*i_x[j]);

            }
        };
#endif




}
