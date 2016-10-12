#include <iostream>
#include <iomanip>
#include "../numerics/assembly.hpp"
#include "../numerics/basis_functions.hpp"
#include "../numerics/assemble_storage.hpp"
#include "../numerics/assembly_base.hpp"
#include "../galerkin_defs.hpp"
//#include "test_assembly.hpp"
//#include "../functors/mass.hpp"
#include "../functors/assembly_functors.hpp"
#include "../functors/linear_solver.hpp"
#include "../functors/matvec.hpp"
#include "../numerics/assemble_storage.hpp"
#include "../numerics/linear_solver.hpp"

using namespace gridtools;
using namespace gdl;
using namespace gdl::enumtype;

typedef gridtools::interval<gridtools::level<0,-1>, gridtools::level<1,-1> > x_interval;

constexpr unsigned int dof_per_dim_0(4);
constexpr unsigned int dof_per_dim_1(4);
constexpr unsigned int dof_per_dim_2(4);

constexpr unsigned int d1=4;
constexpr unsigned int d2=4;
constexpr unsigned int d3=4;
constexpr unsigned int d4=dof_per_dim_0*dof_per_dim_1*dof_per_dim_2;

int main() {


    std::cout<<std::setprecision(10);

    //![storages]
    // Problem matrix
    using matrix_storage_info_t=storage_info< __COUNTER__, layout_tt<3,4> >;
    using matrix_type=storage_t< matrix_storage_info_t >;
    matrix_storage_info_t A_(d1,d2,d3,d4,d4);
    matrix_type A(A_, 0.e0, "A");// This is the problem matrix before reduction of matching contributions from adjacent elements

    // RHS vector
    using rhs_vector_storage_info_t=storage_info< __COUNTER__, layout_tt<3> >;
    using rhs_vector_type=storage_t< rhs_vector_storage_info_t >;
    rhs_vector_storage_info_t b_(d1,d2,d3,d4);
    rhs_vector_type b(b_, 0.e0, "b");// This is the right hand side vector before reduction of matching contributions from adjacent elements

    // RHS test vector
    using rhs_test_vector_storage_info_t=storage_info< __COUNTER__, layout_tt<3> >;
    using rhs_test_vector_type=storage_t< rhs_test_vector_storage_info_t >;
    rhs_test_vector_storage_info_t b_test_(d1,d2,d3,d4);
    rhs_test_vector_type b_test(b_test_, 0.e0, "b_test");// This is the right hand side vector before reduction of matching contributions from adjacent elements

    // Unknowns vector
    using unk_vector_storage_info_t=storage_info< __COUNTER__, layout_tt<3> >;
    //using unk_vector_type=storage_t< unk_vector_storage_info_t >;
    using unk_vector_type = storage<assemble_storage< unk_vector_storage_info_t, dof_per_dim_0, dof_per_dim_1, dof_per_dim_2> >;
    unk_vector_storage_info_t x_(d1,d2,d3,d4);
    unk_vector_type x(x_, 0.e0, "x");// This is the unknowns vector before reduction of matching contributions from adjacent elements

    // Updated unknowns vector
    using unk_upd_vector_storage_info_t=storage_info< __COUNTER__, layout_tt<3> >;
    using unk_upd_vector_type=storage_t< unk_upd_vector_storage_info_t >;
    unk_upd_vector_storage_info_t x_upd_(d1,d2,d3,d4);
    unk_upd_vector_type x_upd(x_upd_, 0.e0, "x_upd");// This is the unknowns vector before reduction of matching contributions from adjacent elements

    // Err vector
    using err_vector_storage_info_t=storage_info< __COUNTER__, layout_tt<3> >;
    using err_vector_type=storage_t< err_vector_storage_info_t >;
    err_vector_storage_info_t err_(d1,d2,d3,d4);
    err_vector_type err(err_, 0.e0, "err");// TODO: remove this storage!

    // Scaled err vector
    using scaled_err_vector_storage_info_t=storage_info< __COUNTER__, layout_tt<3> >;
    using scaled_err_vector_type=storage_t< scaled_err_vector_storage_info_t >;
    scaled_err_vector_storage_info_t scaled_err_(d1,d2,d3,d4);
    scaled_err_vector_type scaled_err(scaled_err_, 0.e0, "scaled_err");// TODO: remove this storage!

    // Step scalar value
    using step_value_storage_info_t=storage_info< __COUNTER__, layout_tt<3> >;
    using step_value_type=storage_t< step_value_storage_info_t >;
    step_value_storage_info_t omega_(d1,d2,d3,d4);
    step_value_type omega(omega_, 2./(1.+8.), "omega");// This is the richardson update step
    //![storages]


    ///// TEST INPUT
    for(uint_t I=1; I<d1-1; I++)
        for(uint_t J=1; J<d2-1; J++)
            for(uint_t K=1; K<d3-1; K++)
                for(uint_t dof_index1=0; dof_index1<d4; dof_index1++)
                    for(uint_t dof_index2=0; dof_index2<d4; dof_index2++)
                    {
//                            A(I,J,K,dof_index1,dof_index2) = I*100 + J*1000 + K*10000 + dof_index1*d4 + dof_index2;
//                            A(I,J,K,dof_index1,dof_index2) = 1.e0*(dof_index1 + 1);
                        if(dof_index1 == dof_index2){
                            if(dof_index1 == 0)
                                A(I,J,K,dof_index1,dof_index2) = 2.e0;
                            else
                                A(I,J,K,dof_index1,dof_index2) = 1.e0;
//                            if(dof_index1 == 1)
//                                A(I,J,K,dof_index1,dof_index2) = 1.e0;
//                            if(dof_index1 == 2)
//                                A(I,J,K,dof_index1,dof_index2) = 1.e0;
                        }
                        else
                            A(I,J,K,dof_index1,dof_index2) = 0.e0;


                    }

    for(uint_t I=1; I<d1-1; I++)
        for(uint_t J=1; J<d2-1; J++)
            for(uint_t K=1; K<d3-1; K++)
                for(uint_t dof_index=0; dof_index<d4; dof_index++)
                {
                    if(dof_index == 0 || dof_index==1)
                        b(I,J,K,dof_index) = 1.e0;
//                    else
//                        b(I,J,K,dof_index) = 0.;
                }

    const double stability_thr(-1.0);
    const double error_thr(1.e-6);
    double stability;
    uint_t max_iter(50);


#if 0
    //![placeholders]
#if 1
    typedef arg<0, matrix_type> p_A;
    typedef arg<1, rhs_vector_type> p_b;
    typedef arg<2, rhs_test_vector_type> p_b_test;
    typedef arg<3, unk_vector_type> p_x;
    typedef arg<4, unk_upd_vector_type> p_x_upd;
    typedef arg<5, err_vector_type> p_err;
    typedef arg<6, scaled_err_vector_type> p_scaled_err;
    typedef arg<7, step_value_type> p_omega;
    typedef boost::mpl::vector<p_A, p_b, p_b_test, p_x, p_x_upd, p_err, p_scaled_err, p_omega> accessor_list;
    domain_type<accessor_list> domain(boost::fusion::make_vector(&A, &b, &b_test, &x, &x_upd, &err, &scaled_err, &omega));
#else
    ////&&&&&&&&&&

    using geo_map=reference_element<2, Lagrange, Hexa>;
    using fe=reference_element<2, Lagrange, Hexa>;
    using cub=cubature<fe::order+1, fe::shape>;
    using geo_t = intrepid::unstructured_geometry<geo_map, cub>;
    using as=assembly<geo_t>;
    using as_base=assembly_base<geo_t>;
    geo_t geo_;

    as assembler(geo_,d1,d2,d3);
    as_base assembler_base(d1,d2,d3);

    using domain_tuple_t = domain_type_tuple< as, as_base>;
    domain_tuple_t domain_tuple_ (assembler, assembler_base);
    // defining the placeholder for the local basis/test functions
    using dt = domain_tuple_t;
    typedef arg<dt::size, matrix_type> p_A;
    typedef arg<dt::size+1, rhs_vector_type> p_b;
    typedef arg<dt::size+2, unk_vector_type> p_x;
    typedef arg<dt::size+3, unk_upd_vector_type> p_x_upd;
    typedef arg<dt::size+4, step_value_type> p_omega;
    auto domain=domain_tuple_.template domain<p_A, p_b, p_x, p_x_upd, p_omega>(A, b, x, x_upd, omega);
#endif
    //![placeholders]

    //![grid]
    auto mesh_coords=grid<axis>({1, 0, 1, d1-1, d1},
                                {1, 0, 1, d2-1, d2});
    mesh_coords.value_list[0] = 1;
    mesh_coords.value_list[1] = d3-1;
    //![grid]


    std::cout<<"Computing disassembled richardson"<<std::endl;

    do{

        //![richardson_iteration]
        auto richardson_iteration=make_computation<BACKEND>(domain,
                                                            mesh_coords,
                                                            make_mss(execute<forward>(),
                                                            make_esf<functors::matvec>(p_A(),p_x(),p_b_test()),
                                                            make_esf<functors::vecvec<functors::sub_operator<double> > >(p_b(),p_b_test(),p_err()),
                                                            make_esf<functors::vecvec<functors::mult_operator<double> > >(p_omega(),p_err(),p_scaled_err()),
                                                            make_esf<functors::vecvec<functors::sum_operator<double> > >(p_x(),p_scaled_err(),p_x_upd())
                                                            ));
        richardson_iteration->ready();
        richardson_iteration->steady();
        richardson_iteration->run();
        richardson_iteration->finalize();
        //![richardson_iteration]


//        for(uint_t I=0; I<d1; I++)
//            for(uint_t J=0; J<d2; J++)
//                for(uint_t K=0; K<d3; K++)
//                    for(uint_t dof_index=0; dof_index<d4; dof_index++)
//                    {
//                        std::cout<<I<<" "<<J<<" "<<K<<" "<<dof_index<<" b_test = "<<b_test(I,J,K,dof_index)<<
//                                      " err "<<err(I,J,K,dof_index)<<
//                                      " x_upd "<<x_upd(I,J,K,dof_index)<<std::endl;
//                    }


        //![check_stability]
        // TODO this is wrong because of duplicated elements!
//        stability = 0.;
//        for(uint_t I=0; I<d1; I++)
//            for(uint_t J=0; J<d2; J++)
//                for(uint_t K=0; K<d3; K++)
//                    for(uint_t dof_index=0; dof_index<d4; dof_index++)
//                    {
//                        stability += (x(I,J,K,dof_index) - x_upd(I,J,K,dof_index))*(x(I,J,K,dof_index) - x_upd(I,J,K,dof_index));
//                    }
        //![check_stability]

//        stability *=0.1;


        //![assemble_update]
        if(1) {
            auto assemble_update=make_computation<BACKEND>(domain,
                                                           mesh_coords,
                                                           make_mss(execute<forward>(),
                                                                    make_esf<functors::tmp_copy_vector<d4> >(p_x_upd(),p_x()),
                                                                    make_esf<functors::hexahedron_vector_assemble<dof_per_dim_0,dof_per_dim_1,dof_per_dim_2> >(p_x_upd(),p_x())
                                                                    ));
            assemble_update->ready();
            assemble_update->steady();
            assemble_update->run();
            assemble_update->finalize();
        }
        //![assemble_and_update]hexahedron_vector_assemble

        std::cout<<"stability "<<stability<<std::endl;
        for(uint_t I=1; I<d1-1; I++)
            for(uint_t J=1; J<d2-1; J++)
                for(uint_t K=1; K<d3-1; K++)
                    for(uint_t dof_index=0; dof_index<d4; dof_index++)
                    {
                        std::cout<<I<<" "<<J<<" "<<K<<" "<<dof_index<<" x = "<<x(I,J,K,dof_index)<<std::endl;
                    }
        std::cout<<std::endl;


    }while(stability>stability_thr);
#endif

#if 1

    // TEST VS ASSEMBLED MATRIX
//    constexpr gridtools::meta_storage_base<static_int<__COUNTER__>,gt::layout_map<2,1,0>,false> indexing_global{(dof_per_dim_0-1)*d1+1, (dof_per_dim_1-1)*d2+1, (dof_per_dim_2-1)*d3+1};
    constexpr storage_info<__COUNTER__,gt::layout_map<2,1,0> > indexing_global{(dof_per_dim_0-1)*d1+1, (dof_per_dim_1-1)*d2+1, (dof_per_dim_2-1)*d3+1};
    constexpr gridtools::uint_t n_dof{indexing_global.template dims<0>()*indexing_global.template dims<1>()*indexing_global.template dims<2>()};

    using dof_map_storage_info_t=storage_info<__COUNTER__, layout_tt<3> >;
    using dof_map_storage_type=storage_t< dof_map_storage_info_t >;
    dof_map_storage_info_t dof_map_(d1,d2,d3,d4);
    dof_map_storage_type dof_map(dof_map_, "dof_map");

    using assembled_matrix_storage_info_t=storage_info<__COUNTER__,layout_tt<> >;
    using assembled_matrix_storage_type=storage_t< assembled_matrix_storage_info_t >;
    assembled_matrix_storage_info_t A_ass_(n_dof,n_dof,1);
    assembled_matrix_storage_type A_ass(A_ass_, 0.e0, "A_ass");

    using assembled_rhs_storage_info_t=storage_info<__COUNTER__,layout_tt<> >;
    using assembled_rhs_storage_type=storage_t< assembled_rhs_storage_info_t >;
    assembled_rhs_storage_info_t b_ass_(n_dof,1,1);
    assembled_rhs_storage_type b_ass(b_ass_, 0.e0, "b_ass");

    using assembled_unk_vector_storage_info_t=storage_info<__COUNTER__, layout_tt<> >;
    using assembled_unk_vector_type=storage_t< assembled_unk_vector_storage_info_t >;
    assembled_unk_vector_storage_info_t x_ass_(n_dof,1,1);
    assembled_unk_vector_type x_ass(x_ass_, 0.e0, "x_ass");

    using assembled_unk_upd_vector_storage_info_t=storage_info<__COUNTER__, layout_tt<> >;
    using assembled_unk_upd_vector_type=storage_t< assembled_unk_upd_vector_storage_info_t >;
    assembled_unk_upd_vector_storage_info_t x_upd_ass_(n_dof,1,1);
    assembled_unk_upd_vector_type x_upd_ass(x_upd_ass_, 0.e0, "x_upd_ass");

    using assembled_omega_storage_info_t=storage_info<__COUNTER__, layout_tt<> >;
    using assembled_omega_type=storage_t< assembled_omega_storage_info_t >;
    assembled_omega_storage_info_t omega_ass_(1,1,1);
    assembled_omega_type omega_ass(omega_ass_, 0.1e0, "omega_ass");

    for(uint_t I=0; I<d1; I++)
        for(uint_t J=0; J<d2; J++)
            for(uint_t K=0; K<d3; K++)
                {
                    uint_t dofIndex=0;
                    for(u_short k=0;k<dof_per_dim_2;++k)
                        for(u_short j=0;j<dof_per_dim_1;++j)
                            for(u_short i=0;i<dof_per_dim_0;++i,++dofIndex)
                                {
                                    dof_map(I,J,K,dofIndex) = i+j*indexing_global.template dims<0>()+k*indexing_global.template dims<0>()*indexing_global.template dims<1>() +
                                            I*(dof_per_dim_0 - 1) + J*indexing_global.template dims<0>()*(dof_per_dim_1-1) + K*indexing_global.template dims<0>()*indexing_global.template dims<1>()*(dof_per_dim_2-1);
                                }
                }



    typedef arg<0, matrix_type> p_A_disass;
    typedef arg<1, rhs_vector_type> p_b_disass;
    typedef arg<2, dof_map_storage_type> p_dof_map;
    typedef arg<3, assembled_matrix_storage_type> p_A_ass;
    typedef arg<4, assembled_rhs_storage_type> p_b_ass;
    typedef boost::mpl::vector<p_A_disass,p_b_disass,p_dof_map,p_A_ass,p_b_ass> accessor_ass;
    domain_type<accessor_ass> domain_ass(boost::fusion::make_vector(&A,&b,&dof_map,&A_ass,&b_ass));

    auto coords_ass=grid<axis>({0, 0, 0, n_dof-1, n_dof},
                               {0, 0, 0, n_dof-1, n_dof});
    coords_ass.value_list[0] = 0;
    coords_ass.value_list[1] = 0;
#if 0

    std::cout<<"Running matrix assemble"<<std::endl;

    auto computation_ass= make_positional_computation<BACKEND >(domain_ass, coords_ass,
                                                                make_mss( execute<forward>(),
                                                                          make_esf<functors::global_assemble_no_if>(p_A_disass(),p_dof_map(),p_A_ass())));

    computation_ass->ready();
    computation_ass->steady();
    computation_ass->run();
    computation_ass->finalize();


    auto coords_ass_vect=grid<axis>({0, 0, 0, n_dof-1, n_dof},
                                    {0, 0, 0, 0, 1});
    coords_ass_vect.value_list[0] = 0;
    coords_ass_vect.value_list[1] = 0;


    std::cout<<"Running vector assemble"<<std::endl;

    auto computation_vect_ass= make_positional_computation<BACKEND >(domain_ass, coords_ass_vect,
                                                                        make_mss( execute<forward>(),
                                                                                  make_esf<functors::global_vector_assemble_no_if>(p_b_disass(),p_dof_map(),p_b_ass())));

    computation_vect_ass->ready();
    computation_vect_ass->steady();
    computation_vect_ass->run();
    computation_vect_ass->finalize();

//    for(uint_t i=0;i<n_dof;++i)
//        std::cout<<"Ass b "<<b_ass(i,0,0)<<std::endl;

    typedef arg<0, assembled_matrix_storage_type> p_A_ass_again;
    typedef arg<1, assembled_rhs_storage_type> p_b_ass_again;
    typedef arg<2, assembled_omega_type> p_omega_ass;
    typedef arg<3, assembled_unk_vector_type> p_x_ass;
    typedef arg<4, assembled_unk_upd_vector_type> p_x_ass_upd;

    typedef boost::mpl::vector<p_A_ass_again, p_b_ass_again, p_omega_ass, p_x_ass, p_x_ass_upd> accessor_list_rich_ass;
    domain_type<accessor_list_rich_ass> domain_ass_rich(boost::fusion::make_vector(&A_ass, &b_ass, &omega_ass, &x_ass, &x_upd_ass));


    constexpr uint_t num_problems = 1;
    auto coords_ass_rich=grid<axis>({0, 0, 0, n_dof-1, n_dof},
                                    {0, 0, 0, num_problems-1, num_problems});
    coords_ass_rich.value_list[0] = 0;
    coords_ass_rich.value_list[1] = 0;

    std::cout<<"Computing assembled richardson"<<std::endl;

    uint_t num_iterations(0);
    do {
        num_iterations++;
        auto iterate= make_computation<BACKEND >(domain_ass_rich, coords_ass_rich,
                make_mss( execute<forward>(),
                          make_esf<functors::richardson_iteration>(p_A_ass_again(), p_b_ass_again(), p_omega_ass(), p_x_ass(), p_x_ass_upd())));

        iterate->ready();
        iterate->steady();
        iterate->run();
        iterate->finalize();

        stability = 0.;
        for(uint_t row = 0;row<n_dof;++row) {
                stability+=(x_upd_ass(row,0,0) - x_ass(row,0,0))*(x_upd_ass(row,0,0) - x_ass(row,0,0));
        }

//        std::cout<<"stability ass "<<stability<<std::endl;



        for(uint_t i=0;i<n_dof;++i){
                x_ass(i,0,0) = x_upd_ass(i,0,0);
        }

//        std::cout<<std::endl;

    }while(stability>stability_thr && num_iterations<max_iter);

#endif
#if 1
    //    float_t* ptr_A = new float_t[25];
    //    float_t* ptr_b = new float_t[5];
    //    float_t* ptr_x  = new float_t[5];
        //    iterative_linear_solver<richardson_iteration>::solve(5, ptr_A, ptr_b, ptr_x, new_omega, max_iter);
//    iterative_linear_solver< gdl::richardson_iteration<dof_per_dim_0,dof_per_dim_1,dof_per_dim_2> >::solve(A, b, x, stability_thr, max_iter);
    linear_solver< gdl::cg_solver<dof_per_dim_0,dof_per_dim_1,dof_per_dim_2> >::solve(A, b, x, stability_thr, error_thr, max_iter);
#endif


    for(uint_t row = 0;row<n_dof;++row) {

//        std::cout<<row<<" "<<x_ass(row,0,0)<<" "<<x.get_value(row)<<std::endl;

        if(x_ass(row,0,0)!=x.get_value(row))
            {
                std::cout<<"ERROR! "<<row<<" "<<x_ass(row,0,0)<<" "<<x.get_value(row)<<std::endl;
            }

    }

//    std::cout<<"A matr"<<std::endl;
//
//    for(uint_t row = 0;row<n_dof;++row) {
//            for(uint_t col = 0;col<n_dof;++col) {
//
//                    if(A_ass(row,col,0)!=0)
//                        std::cout<<row<<" "<<col<<" "<<A_ass(row,col,0)<<std::endl;
//            }
//    }



//    using matrix_storage_info_t=storage_info< __COUNTER__, layout_tt<3,4> >;
//    using matrix_type=storage_t< matrix_storage_info_t >;
//    matrix_storage_info_t A_(d1,d2,d3,d4,d4);
//    matrix_type A(A_, 0., "A");//
//
//    template <uint_t Counter_A, uint_t Counter_b, uint_t Counter_x, typename ... Params>
//    static inline void solve(linear_solver_stor_defs::matrix_storage_t<Counter_A> const & i_A,
//             linear_solver_stor_defs::vector_storage_t<Counter_b> const & i_b,
//             linear_solver_stor_defs::vector_storage_t<Counter_x> const & io_x,
//             Params const & ... i_params)
//
//
//
//    template <uint_t Counter>
//    using matrix_storage_info_t = storage_info< Counter, layout_tt<3,4> >;
//
//    template <uint_t Counter>
//    using matrix_storage_t = storage_t< matrix_storage_info_t<Counter> >;



#endif

    return 0;
}
