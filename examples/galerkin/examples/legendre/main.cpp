#include "legendre.hpp"
#include "boundary.hpp"
#include "compute_jacobian.hpp"
#include "compute_assembly.hpp"
#include "../../tools/io.hpp"

namespace gdl{

    using namespace gt::expressions;

    struct bc_functor{

        using bc=gt::accessor<0, enumtype::in, gt::extent<> , 4>;
        using out=gt::accessor<1, enumtype::inout, gt::extent<> , 4>;
        using arg_list=boost::mpl::vector<bc, out> ;

        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_interval) {
            gt::dimension<4> I;

            uint_t const n_points=eval.template get_storage_dim<3>(bc());

            //assign the points on the boundary layer of elements
            for(int i=0; i<n_points; ++i){
                eval(out(I+i)) = eval(bc(I+i));
            }
        }
    };
}

using namespace legendre;

int main( int argc, char ** argv){

    if(argc!=5) {
        printf("usage: \n >> legendre <d1> <d2> <d3> <N>\n");
        exit(-666);
    }
    int it_ = atoi(argv[4]);

    using namespace gdl;
    using namespace gdl::enumtype;

    uint_t d1= atoi(argv[1]);
    uint_t d2= atoi(argv[2]);
    uint_t d3= atoi(argv[3]);

    using bc_storage_info_t=storage_info< __COUNTER__, gt::layout_map< -1,0,1,2 > >;
    using bc_storage_t = storage_t< bc_storage_info_t >;

    using bc_tr_storage_info_t=storage_info< __COUNTER__, gt::layout_map< -1,0,1,2 > >;
    using bc_tr_storage_t = storage_t< bc_tr_storage_info_t >;

    bc_storage_info_t bc_low_meta_(1,d2,d3, bd_geo_cub_t::bd_cub::numCubPoints());
    bc_storage_t bc_low_(bc_low_meta_, 0.);

    bc_tr_storage_info_t tr_bc_low_meta_(1,d2,d3, discr_map::basis_cardinality());
    bc_tr_storage_t tr_bc_low_(tr_bc_low_meta_, 0.);

    mesh mesh_(d1,d2,d3);
    legendre_advection problem_(// mesh_,
        d1,d2,d3);

    compute_jacobian jac_(problem_);
    jac_.assemble();

    compute_assembly ass_(problem_);
    ass_.assemble();


    // //removing unused storage
    //assembler_base.grid().release();
    problem_.assembler().jac().release();
    // assembler.fe_backend().cub_weights().release();
    // assembler.jac_det().release();
    problem_.assembler().jac_inv().release();
    // assembler.fe_backend().val().release();
    // assembler.fe_backend().grad().release();
    problem_.bd_assembler().bd_jac().release();
    problem_.bd_assembler().normals().release();
    // bd_assembler.bd_measure().release();
    // bd_assembler.bd_backend().bd_cub_weights().release();
    problem_.bd_assembler().bd_backend().ref_normals().release();
    // bd_assembler.bd_backend().val().release();
    // bd_assembler.bd_backend().grad().release();
    problem_.beta_phys().release();
    // geo_.val().release();
    // geo_.grad().release();
    problem_.normals().release();

    // bc_apply< as, discr_t, bc_functor, functors::upwind> bc_apply_(assembler, fe_);
    // auto bc_compute_low = bc_apply_.compute(coords_low, bc_low_, tr_bc_low_);
    // auto bc_apply_low = bc_apply_.template apply(coords_low
    //                                              ,tr_bc_low_
    //                                              ,result_
    //                                              ,bd_beta_n_
    //                                              ,bd_mass_
    //                                              ,bd_mass_uv_
    //                                              ,u_
    //     );

    // boundary condition computation

    struct transform_bc {
        typedef  gt::arg<0, typename as::storage_type >    p_jac_det;
        typedef  gt::arg<1, typename as::geometry_t::weights_storage_t >   p_weights;
        typedef  gt::arg<2, typename discr_t::basis_function_storage_t> p_phi;
        typedef  gt::arg<3, bc_storage_t > p_bc;
        typedef  gt::arg<4, bc_tr_storage_t > p_bc_integrated;
    };

    typedef typename boost::mpl::vector< typename transform_bc::p_jac_det, typename transform_bc::p_weights, typename transform_bc::p_phi, typename transform_bc::p_bc, typename transform_bc::p_bc_integrated> mpl_list_transform_bc;

    gt::aggregator_type<mpl_list_transform_bc> domain_transform_bc(
        boost::fusion::make_vector(
            &problem_.assembler().jac_det()
            ,&problem_.assembler().fe_backend().cub_weights()
            ,&problem_.fe().val()
            ,&bc_low_
            ,&tr_bc_low_
            ) );

    auto coords_low=gt::grid<axis>({1u,0u,1u,d1-1u,d1},
        {1u, 0u, 1u, (uint_t)d2-1u, (uint_t)d2});
    coords_low.value_list[0] = 0;
    coords_low.value_list[1] = 0;


    auto bc_compute_low=gt::make_computation< BACKEND >(
        domain_transform_bc, coords_low
        , gt::make_multistage(
            execute<forward>()
            , gt::make_stage< bc_functor >( typename transform_bc::p_bc(), typename transform_bc::p_bc() )
            , gt::make_stage< functors::transform >( typename transform_bc::p_jac_det(), typename transform_bc::p_weights(), typename transform_bc::p_phi(), typename transform_bc::p_bc(), typename transform_bc::p_bc_integrated() )
            )
        );

    struct bc{
        typedef  gt::arg<0, bc_tr_storage_t > p_bc;
        typedef  gt::arg<1, scalar_type > p_result;
        typedef  gt::arg<2, bd_scalar_type > p_beta_n;
        typedef  gt::arg<3, bd_matrix_type > p_bd_mass_uu;
        typedef  gt::arg<4, bd_matrix_type > p_bd_mass_uv;
    };

    typedef typename boost::mpl::vector< typename bc::p_bc, typename bc::p_result, typename bc::p_beta_n, typename bc::p_bd_mass_uu, typename bc::p_bd_mass_uv> mpl_list_bc;

    gt::aggregator_type<mpl_list_bc> domain_apply_bc(boost::fusion::make_vector(
                                                     &tr_bc_low_
                                                     ,&problem_.result()
                                                     ,&problem_.bd_beta_n()
                                                     ,&problem_.bd_mass()
                                                     ,&problem_.bd_mass_uv()
                                                     ));

    auto bc_apply_low=gt::make_computation< BACKEND >(
        domain_apply_bc, coords_low
        , gt::make_multistage(
            execute<forward>()
            , gt::make_stage< functors::upwind2 >(typename bc::p_bc(), typename bc::p_beta_n(), typename bc::p_bd_mass_uu(), typename bc::p_bd_mass_uv(),  typename bc::p_result())
            )
        );



    //initialization of the boundary condition
    for(uint_t j=0; j<d2; ++j)
        for(uint_t k=0; k<d3; ++k)
            for(uint_t dof=0; dof<bd_geo_cub_t::bd_cub::numCubPoints(); ++dof)
            {
                if(j<d2+1)
                    bc_low_(666, j, k, dof) = 1.;
                else
                    bc_low_(666, j, k, dof) = 0.;
                // bc_right_( j, 666, k, dof) = 1.;
            }

    /* end of boundary conditions */

    int n_it_ = it_;

    // gt::io_rectilinear_qpoints< as_base_t::grid_type, discr_t::cub_points_storage_t, gt::static_ushort<cub::cubDegree> > io_(problem_.grid(), problem_.fe().get_cub_points());

    bc_compute_low->ready();
    bc_apply_low->ready();
    bc_compute_low->steady();
    bc_apply_low->steady();

    for(int i=0; i<n_it_; ++i){ // Richardson iterations
        bc_compute_low->run();
        bc_apply_low->run();
        problem_.run();
    }
    bc_compute_low->run();
    bc_apply_low->run();

    // io_.set_information("Time");
    // io_.set_attribute_scalar<0>(result_, "Ax");
    // io_.set_attribute_scalar<0>(u_, "solution");
    // io_.set_attribute_vector_on_face<0>(face_vec, "normals");
    // io_.write("grid");

    // for(int i=0; i<d1; ++i)
    //     for(int j=0; j<d2; ++j)
    //         for(int k=0; k<d3; ++k)
    //             for(int l=0; l<10; ++l)
    //                 std::cout<< mass_(i,j,k,l,l)<<"\n";

    // spy(mass_, "mass.txt");
//     spy_vec(result_, "sol.txt");
//     spy_vec(u_, "init.txt");
    problem_.eval();

    problem_.finalize();
    bc_compute_low->finalize();
    bc_apply_low->finalize();

#ifndef CUDA_EXAMPLE
    std::cout << bc_compute_low->print_meter() << std::endl;
    std::cout << bc_apply_low->print_meter() << std::endl;
#endif
    // io_.set_attribute_scalar<0>(problem_.result_interpolated(), "solution");

    // spy(advection_, "advection.txt");
}
