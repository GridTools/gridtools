// TODO: clean include list, assembly.hpp should not be here
#include <iostream>
#include <iomanip>
#include "../numerics/assembly.hpp"
#include "../numerics/assemble_storage.hpp"
#include "../numerics/linear_solver.hpp"
#include "../functors/stiffness.hpp"

using namespace gridtools;
using namespace gdl;
using namespace gdl::enumtype;


namespace gdl {

    namespace functors {

        typedef gridtools::interval<gridtools::level<0,-1>, gridtools::level<1,-1> > x_interval;

        struct test_fun {

            using in1=gt::accessor<0, enumtype::in, gt::extent<> , 5> ;
//            using in2=gt::accessor<1, enumtype::in, gt::extent<> , 4> ;
            using out=gt::accessor<1, enumtype::inout, gt::extent<> , 5> ;
//            using arg_list=boost::mpl::vector< in1, in2, out > ;
            using arg_list=boost::mpl::vector< in1, out > ;

            template <typename Evaluation>
            GT_FUNCTION
            static void Do(Evaluation const & eval, x_interval) {
                gt::dimension<4> row;
                uint_t const num_rows=eval.get().template get_storage_dims<3>(in1());

                // Loop over vector elements
                for(uint_t i=0;i<num_rows;++i){
//                    eval(out(row+i)) = eval(in1(row+i)) + eval(in2(row+i));
                }
            }

        };


    }
}

namespace gdl {

    namespace functors {

        template <ushort_t N_DOF0, ushort_t N_DOF1, ushort_t N_DOF2>
        struct set_value {

            using inout=gt::accessor<0, enumtype::inout, gt::extent<> , 4> ;
            using arg_list=boost::mpl::vector< inout > ;

            template <typename Evaluation>
            GT_FUNCTION
            static void Do(Evaluation const & eval, x_interval) {
                gt::dimension<4> dof;

                constexpr gt::meta_storage_base<static_int<__COUNTER__>,gt::layout_map<2,1,0>,false> indexing{N_DOF0,N_DOF1,N_DOF2};

                // 1 A
                for(short_t I1=1; I1<indexing.template dims<0>()-1; I1++)
                    for(short_t J1=1; J1<indexing.template dims<1>()-1; J1++)
                    {

                        eval(inout(dof+indexing.index(I1,J1,0))) = 1.e0;

                        eval(inout(dof+indexing.index(J1,0,I1))) = 1.e0;

                        eval(inout(dof+indexing.index(0,I1,J1))) = 1.e0;
                    }

                // 2 B
                short_t J1=0;
                for(short_t I1=1; I1<indexing.template dims<0>()-1; I1++)
                {

                    eval(inout(dof+indexing.index(I1,J1,0))) = 1.e0;

                    eval(inout(dof+indexing.index(J1,0,I1))) = 1.e0;

                    eval(inout(dof+indexing.index(0,I1,J1))) = 1.e0;

                }

                // 3 F
                eval(inout(dof+0)) = 1.e0;

            }

        };

    }
}


#define PRINT_DATA_FILES false

int main(){

    //![solver_setup]
    const double stability_thr(-1.0);
    const double error_thr(1.e-20);
    double stability;
    uint_t max_iter(1);
    //![solver_setup]

    //![domain_definitions]
    constexpr float_t dx{2.};
    constexpr float_t dy{2.};
    constexpr float_t dz{2.};
    constexpr uint_t d1=2;
    constexpr uint_t d2=1;
    constexpr uint_t d3=1;
    //![domain_definitions]

    //![FEM_definitions]
    using fe=reference_element<1, Lagrange, Hexa>;
    using geo_map=reference_element<1, Lagrange, Hexa>;
    using cub=cubature<fe::order()+1, fe::shape()>;
    using geo_t = intrepid::unstructured_geometry<geo_map, cub>;
    using discr_t = intrepid::discretization<fe, cub>;
    using as=assembly<geo_t>;
    using as_base=assembly_base<geo_t>;
    constexpr uint_t dof_per_dim{2};//TODO: compute this information from FEM traits
    constexpr uint_t dof_per_el{dof_per_dim*dof_per_dim*dof_per_dim};//TODO: compute this information from FEM traits
    constexpr gridtools::uint_t n_dof{((dof_per_dim-1)*d1+1)*((dof_per_dim-1)*d2+1)*((dof_per_dim-1)*d3+1)};
    constexpr gridtools::uint_t n_dofx{(dof_per_dim-1)*d1+1};
    constexpr gridtools::uint_t n_dofy{(dof_per_dim-1)*d2+1};
    constexpr gridtools::uint_t n_dofz{(dof_per_dim-1)*d3+1};
    //![FEM_definitions]

    //![storages_definitions]
    // Stiffness matrix
    using stiffness_matrix_storage_info_t=storage_info<  __COUNTER__, layout_tt<3,4> >;
//    using stiffness_matrix_type = storage_t< stiffness_matrix_storage_info_t >;
    using stiffness_matrix_type = storage<assemble_storage< stiffness_matrix_storage_info_t, dof_per_dim, dof_per_dim, dof_per_dim> >;//TODO: default storage needed (unassembled content)
    // Source vector
    using source_vector_storage_info_t=storage_info<  __COUNTER__, layout_tt<3> >;
    using source_vector_type = storage<assemble_storage< source_vector_storage_info_t, dof_per_dim, dof_per_dim, dof_per_dim> >;//TODO: default storage needed (unassembled content)
    // Solution vector
    using solution_vector_storage_info_t=storage_info< __COUNTER__, layout_tt<3> >;
    using solution_vector_type = storage<assemble_storage< solution_vector_storage_info_t, dof_per_dim, dof_per_dim, dof_per_dim> >;
    // Dirichlet bc stiffness mask
    using dirichlet_mask_matrix_storage_info_t=storage_info< __COUNTER__, layout_tt<3,4> >;
    using dirichlet_mask_matrix_type = storage<assemble_storage< dirichlet_mask_matrix_storage_info_t, dof_per_dim, dof_per_dim, dof_per_dim> >;
    // Dirichlet bc source mask
    using dirichlet_mask_vector_storage_info_t=storage_info< __COUNTER__, layout_tt<3> >;
    using dirichlet_mask_vector_type = storage<assemble_storage< dirichlet_mask_vector_storage_info_t, dof_per_dim, dof_per_dim, dof_per_dim> >;
    //![storages_definitions]

    //![storages_allocation]
    // Computational domain
    constexpr uint_t comp_d1{d1+2};
    constexpr uint_t comp_d2{d2+2};
    constexpr uint_t comp_d3{d3+2};
    // Stiffness matrix
    stiffness_matrix_storage_info_t stiffness_(comp_d1,comp_d2,comp_d3,dof_per_el,dof_per_el);
    stiffness_matrix_type stiffness(stiffness_, halo_data(2,2,2,1,1,1), 0.e0, "stiffness");// This is the unassembled problem matrix
    // Source vector
    source_vector_storage_info_t source_(comp_d1,comp_d2,comp_d3,dof_per_el);
    source_vector_type source(source_, halo_data(2,2,2,1,1,1), 0.e0, "source");// This is the unassembled right hand side vector
    // Solution vector
    solution_vector_storage_info_t sol_(comp_d1,comp_d2,comp_d3,dof_per_el);
    solution_vector_type sol(sol_, halo_data(2,2,2,1,1,1), 0.e0, "sol");// This is the assembled unknowns vector
    // Dirichlet bc stiffness mask
    dirichlet_mask_matrix_storage_info_t dirichlet_bc_mask_matrix_(comp_d1,comp_d2,comp_d3,dof_per_el,dof_per_el);
    dirichlet_mask_matrix_type dirichlet_bc_mask_matrix(dirichlet_bc_mask_matrix_, halo_data(2,2,2,1,1,1), 0.e0, "dirichlet_bc_mask_matrix");// This is the dof mask for dirichlet bc application to stiffness matrix
    // Dirichlet bc source mask
    dirichlet_mask_vector_storage_info_t dirichlet_bc_mask_vector_(comp_d1,comp_d2,comp_d3,dof_per_el);
    dirichlet_mask_vector_type dirichlet_bc_mask_vector(dirichlet_bc_mask_vector_, halo_data(2,2,2,1,1,1), 0.e0, "dirichlet_bc_mask_vector");// This is the dof mask for dirichlet bc application to source vector
    //![storages_allocation]


    ///////// TODO: temporary solution for problem data definition
    //![source_definition]
    //![placeholders]
    typedef arg<0, source_vector_type> p_source;
    //![placeholders]

    //![domain]
    typedef boost::mpl::vector<p_source> source_set_domain_accessors;
    gridtools::domain_type<source_set_domain_accessors> source_set_domain(boost::fusion::make_vector(&source));


    auto set_source_coords=grid<axis>({1, 0, 1, d1, d1+1},
                                      {1, 0, 1, d2, d2+1});
    set_source_coords.value_list[0] = 1;
    set_source_coords.value_list[1] = d3;
    //![domain]

    //![computation]
    auto computation=gt::make_computation<BACKEND>(
        source_set_domain,
        set_source_coords,
        make_mss(
            execute<forward>(),
            gt::make_esf<functors::set_value<dof_per_dim,dof_per_dim,dof_per_dim> >( p_source() ),
            gt::make_esf<functors::hexahedron_vector_distribute<dof_per_dim,dof_per_dim,dof_per_dim> >(p_source())
        ));

    computation->ready();
    computation->steady();
    computation->run();
    computation->finalize();
    //![computation]
    //![source_definition]
    ////////////////////////////////////////////////////////////



    //![stiffness_calculation]

    //![as_instantiation]
    geo_t geo_;
    discr_t fe_;
    geo_.compute(Intrepid::OPERATOR_GRAD);//redundants
    geo_.compute(Intrepid::OPERATOR_VALUE);
    fe_.compute(Intrepid::OPERATOR_GRAD);
    fe_.compute(Intrepid::OPERATOR_VALUE);
    constexpr uint_t ass_d1{d1+1};
    constexpr uint_t ass_d2{d2+1};
    constexpr uint_t ass_d3{d3+1};
    as assembler( geo_, ass_d1, ass_d2, ass_d3);
    as_base assembler_base(ass_d1, ass_d2, ass_d3);
    //![as_instantiation]


    //![grid]
    //constructing a structured cartesian grid
    for (uint_t i=0; i<ass_d1; i++)
        for (uint_t j=0; j<ass_d2; j++)
            for (uint_t k=0; k<ass_d3; k++){
                uint_t point = 0;
                for(uint_t dofx = 0;dofx<dof_per_dim;++dofx)
                    for(uint_t dofy = 0;dofy<dof_per_dim;++dofy)
                        for(uint_t dofz = 0;dofz<dof_per_dim;++dofz,++point){
                            assembler_base.grid()( i,  j,  k,  point,  0)= i*(dof_per_dim-1)*dx + dofx*dx;
                            assembler_base.grid()( i,  j,  k,  point,  1)= j*(dof_per_dim-1)*dy + dofy*dy;
                            assembler_base.grid()( i,  j,  k,  point,  2)= k*(dof_per_dim-1)*dz + dofz*dz;
                        }
            }
    //![grid]


    //![placeholders]
    typedef arg<0, as_base::grid_type> p_grid_points;
    typedef arg<1, discr_t::grad_storage_t> p_dphi;
    typedef arg<2, discr_t::grad_storage_t> p_dpsi;
    typedef arg<3, as::jacobian_type> p_jac;
    typedef arg<4, as::storage_type> p_jac_det;
    typedef arg<5, as::jacobian_type> p_jac_inv;
    typedef arg<6, as::weights_storage_t> p_weights;
    typedef arg<7, stiffness_matrix_type> p_stiffness;
    //![placeholders]

    //![domain]
    typedef boost::mpl::vector<p_grid_points, p_dphi, p_dpsi, p_jac, p_jac_det, p_jac_inv, p_weights, p_stiffness> domain_accessors;
    gridtools::domain_type<domain_accessors> domain(boost::fusion::make_vector(&assembler_base.grid(), &fe_.grad(), &fe_.grad(), &assembler.jac(), &assembler.jac_det(), &assembler.jac_inv(), &assembler.cub_weights(), &stiffness));


    auto coords=grid<axis>({1, 0, 1, ass_d1-1, ass_d1},
                           {1, 0, 1, ass_d2-1, ass_d2});
    coords.value_list[0] = 1;
    coords.value_list[1] = ass_d3-1;
    //![domain]

    //![computation]
    computation=gt::make_computation<BACKEND>(
        domain,
        coords,
        make_mss(
            execute<forward>(),
            gt::make_esf<functors::update_jac<geo_t> >( p_grid_points(), p_dphi(), p_jac()),
            gt::make_esf<functors::det<geo_t> >(p_jac(), p_jac_det()),
            gt::make_esf<functors::inv<geo_t> >(p_jac(), p_jac_det(), p_jac_inv()),
            gt::make_esf<functors::stiffness<fe, cub> >(p_jac_det(), p_jac_inv(), p_weights(), p_stiffness(), p_dphi(), p_dpsi())//stiffness
        ));

    computation->ready();
    computation->steady();
    computation->run();
    computation->finalize();
    //![computation]

    //![stiffness_calculation]

    //![boundary_condition_application]

    // x=start
    uint dof1_x=0;
    uint dof2_x=0;
    uint dof1_y=0;
    uint dof2_y=0;
    uint dof1_z=0;
    uint dof2_z=0;
    for(dof1_y=0;dof1_y<n_dofy;++dof1_y)
        for(dof1_z=0;dof1_z<n_dofz;++dof1_z){
            const uint_t dof1 = dof1_x + dof1_y*n_dofx + dof1_z*n_dofx*n_dofy;
//            dirichlet_bc_mask_vector.set_value(dof1) = 0.;
            source.set_value(dof1) = 0.;
            for(dof2_x=0;dof2_x<n_dofy;++dof2_x)
                for(dof2_y=0;dof2_y<n_dofy;++dof2_y)
                    for(dof2_z=0;dof2_z<n_dofz;++dof2_z){
                        const uint_t dof2 = dof2_x + dof2_y*n_dofx + dof2_z*n_dofx*n_dofy;
                        if(dof1!=dof2) {
//                            dirichlet_bc_mask_matrix.set_value(dof1,dof2) = 0.;
//                            dirichlet_bc_mask_matrix.set_value(dof2,dof1) = 0.;
                            stiffness.set_value(dof1,dof2) = 0.;
                            stiffness.set_value(dof2,dof1) = 0.;
                        }
                        else {
                            stiffness.set_value(dof1,dof2) = 1.;
                        }
                    }
        }

    // x=stop
    dof1_x=n_dofx-1;
    for(dof1_y=0;dof1_y<n_dofy;++dof1_y)
        for(dof1_z=0;dof1_z<n_dofz;++dof1_z){
            const uint_t dof1 = dof1_x + dof1_y*n_dofx + dof1_z*n_dofx*n_dofy;
//            dirichlet_bc_mask_vector.set_value(dof1) = 0.;
            source.set_value(dof1) = 0.;
            for(dof2_x=0;dof2_x<n_dofy;++dof2_x)
                for(dof2_y=0;dof2_y<n_dofy;++dof2_y)
                    for(dof2_z=0;dof2_z<n_dofz;++dof2_z){
                        const uint_t dof2 = dof2_x + dof2_y*n_dofx + dof2_z*n_dofx*n_dofy;
                        if(dof1!=dof2) {
//                            dirichlet_bc_mask_matrix.set_value(dof1,dof2) = 0.;
//                            dirichlet_bc_mask_matrix.set_value(dof2,dof1) = 0.;
                            stiffness.set_value(dof1,dof2) = 0.;
                            stiffness.set_value(dof2,dof1) = 0.;
                        }
                        else {
                            stiffness.set_value(dof1,dof2) = 1.;
                        }
                    }
        }


    // y=start
    dof1_y=0;
    for(dof1_x=0;dof1_x<n_dofx;++dof1_x)
        for(dof1_z=0;dof1_z<n_dofz;++dof1_z){
            const uint_t dof1 = dof1_x + dof1_y*n_dofx + dof1_z*n_dofx*n_dofy;
//            dirichlet_bc_mask_vector.set_value(dof1) = 0.;
            source.set_value(dof1) = 0.;
            for(dof2_x=0;dof2_x<n_dofy;++dof2_x)
                for(dof2_y=0;dof2_y<n_dofy;++dof2_y)
                    for(dof2_z=0;dof2_z<n_dofz;++dof2_z){
                        const uint_t dof2 = dof2_x + dof2_y*n_dofx + dof2_z*n_dofx*n_dofy;
                        if(dof1!=dof2) {
//                            dirichlet_bc_mask_matrix.set_value(dof1,dof2) = 0.;
//                            dirichlet_bc_mask_matrix.set_value(dof2,dof1) = 0.;
                            stiffness.set_value(dof1,dof2) = 0.;
                            stiffness.set_value(dof2,dof1) = 0.;
                        }
                        else {
                            stiffness.set_value(dof1,dof2) = 1.;
                        }
                    }
        }

    // y=stop
    dof1_y=n_dofy-1;
    for(dof1_x=0;dof1_x<n_dofx;++dof1_x)
        for(dof1_z=0;dof1_z<n_dofz;++dof1_z){
            const uint_t dof1 = dof1_x + dof1_y*n_dofx + dof1_z*n_dofx*n_dofy;
//            dirichlet_bc_mask_vector.set_value(dof1) = 0.;
            source.set_value(dof1) = 0.;
            for(dof2_x=0;dof2_x<n_dofy;++dof2_x)
                for(dof2_y=0;dof2_y<n_dofy;++dof2_y)
                    for(dof2_z=0;dof2_z<n_dofz;++dof2_z){
                        const uint_t dof2 = dof2_x + dof2_y*n_dofx + dof2_z*n_dofx*n_dofy;
                        if(dof1!=dof2) {
//                            dirichlet_bc_mask_matrix.set_value(dof1,dof2) = 0.;
//                            dirichlet_bc_mask_matrix.set_value(dof2,dof1) = 0.;
                            stiffness.set_value(dof1,dof2) = 0.;
                            stiffness.set_value(dof2,dof1) = 0.;
                        }
                        else {
                            stiffness.set_value(dof1,dof2) = 1.;
                        }
                    }
        }

    // z=start
    dof1_z=0;
    for(dof1_x=0;dof1_x<n_dofx;++dof1_x)
        for(dof1_y=0;dof1_y<n_dofy;++dof1_y){
            const uint_t dof1 = dof1_x + dof1_y*n_dofx + dof1_z*n_dofx*n_dofy;
//            dirichlet_bc_mask_vector.set_value(dof1) = 0.;
            source.set_value(dof1) = 0.;
            for(dof2_x=0;dof2_x<n_dofy;++dof2_x)
                for(dof2_y=0;dof2_y<n_dofy;++dof2_y)
                    for(dof2_z=0;dof2_z<n_dofz;++dof2_z){
                        const uint_t dof2 = dof2_x + dof2_y*n_dofx + dof2_z*n_dofx*n_dofy;
                        if(dof1!=dof2) {
//                            dirichlet_bc_mask_matrix.set_value(dof1,dof2) = 0.;
//                            dirichlet_bc_mask_matrix.set_value(dof2,dof1) = 0.;
                            stiffness.set_value(dof1,dof2) = 0.;
                            stiffness.set_value(dof2,dof1) = 0.;
                        }
                        else {
                            stiffness.set_value(dof1,dof2) = 1.;
                        }
                    }
        }

    // z=stop
    dof1_z=n_dofz-1;
    for(dof1_x=0;dof1_x<n_dofx;++dof1_x)
        for(dof1_y=0;dof1_y<n_dofy;++dof1_y){
            const uint_t dof1 = dof1_x + dof1_y*n_dofx + dof1_z*n_dofx*n_dofy;
//            dirichlet_bc_mask_vector.set_value(dof1) = 0.;
            source.set_value(dof1) = 0.;
            for(dof2_x=0;dof2_x<n_dofy;++dof2_x)
                for(dof2_y=0;dof2_y<n_dofy;++dof2_y)
                    for(dof2_z=0;dof2_z<n_dofz;++dof2_z){
                        const uint_t dof2 = dof2_x + dof2_y*n_dofx + dof2_z*n_dofx*n_dofy;
                        if(dof1!=dof2) {
//                            dirichlet_bc_mask_matrix.set_value(dof1,dof2) = 0.;
//                            dirichlet_bc_mask_matrix.set_value(dof2,dof1) = 0.;
                            stiffness.set_value(dof1,dof2) = 0.;
                            stiffness.set_value(dof2,dof1) = 0.;
                        }
                        else {
                            stiffness.set_value(dof1,dof2) = 1.;
                        }
                    }
        }

//    //![bc_application_placeholders]
//    typedef arg<0, dirichlet_mask_matrix_type> p_dirichlet_mask_matrix;
//    typedef arg<1, dirichlet_mask_vector_type> p_dirichlet_mask_vector;
//    typedef arg<2, stiffness_matrix_type> p_stiffness_in;
//    typedef arg<3, source_vector_type> p_source_in;
//    typedef arg<4, stiffness_matrix_type> p_stiffness_bc_applied;
//    typedef arg<5, source_vector_type> p_source_bc_applied;
//    //![bc_application_placeholders]
//
//
//    //![bc_application_domain]
//    typedef boost::mpl::vector<p_dirichlet_mask_matrix, p_dirichlet_mask_vector, p_stiffness_in, p_source_in, p_stiffness_bc_applied, p_source_bc_applied> bc_application_domain_accessors;
//    gridtools::domain_type<bc_application_domain_accessors> bc_application_domain(boost::fusion::make_vector(&dirichlet_bc_mask_matrix, &dirichlet_bc_mask_vector, &stiffness, &source, &stiffness, &source));
//
//    auto bc_preparation_coords=grid<axis>({2, 0, 2, d1, d1+1},
//                                          {2, 0, 2, d2, d2+1});
//    bc_preparation_coords.value_list[0] = 2;
//    bc_preparation_coords.value_list[1] = d3;
//    //![bc_application_domain]

//    //![bc_application_computation]
//    computation=gt::make_computation<BACKEND>(
//        bc_application_domain,
//        bc_preparation_coords,
//        make_mss(
//            execute<forward>(),
//            gridtools::make_esf<functors::hexahedron_matrix_distribute<dof_per_dim,dof_per_dim,dof_per_dim> >(p_dirichlet_mask_matrix()),
//            gridtools::make_esf<functors::hexahedron_vector_distribute<dof_per_dim,dof_per_dim,dof_per_dim> >(p_dirichlet_mask_vector())
//        ));
//
//    computation->ready();
//    computation->steady();
//    computation->run();
//    computation->finalize();


//    auto bc_application_coords=grid<axis>({1, 0, 1, d1, d1+1},
//                                          {1, 0, 1, d2, d2+1});
//    bc_application_coords.value_list[0] = 1;
//    bc_application_coords.value_list[1] = d3;
//    //![bc_application_domain]
//
//    //![bc_application_computation]
//    computation=gt::make_computation<BACKEND>(
//        bc_application_domain,
//        bc_application_coords,
//        make_mss(
//            execute<forward>(),
//            gridtools::make_esf<functors::vecvec<5,functors::mult_operator<float_t> > >(p_dirichlet_mask_matrix(),p_stiffness_in(),p_stiffness_bc_applied()),
//            gridtools::make_esf<functors::vecvec<4,functors::mult_operator<float_t> > >(p_dirichlet_mask_vector(),p_source_in(),p_source_bc_applied())
//        ));
//
//    computation->ready();
//    computation->steady();
//    computation->run();
//    computation->finalize();
//
//
//    //![bc_application_computation]

    //![bc_application_placeholders]
    typedef arg<0, stiffness_matrix_type> p_stiffness_bc;
    typedef arg<1, source_vector_type> p_source_bc;
    //![bc_application_placeholders]

    //![bc_application_domain]
    typedef boost::mpl::vector<p_stiffness_bc, p_source_bc> bc_application_domain_accessors;
    gridtools::domain_type<bc_application_domain_accessors> bc_application_domain(boost::fusion::make_vector(&stiffness, &source));

    auto bc_preparation_coords=grid<axis>({2, 0, 2, d1+1, d1},
                                          {2, 0, 2, d2+1, d2});
    bc_preparation_coords.value_list[0] = 2;
    bc_preparation_coords.value_list[1] = d3+1;
    //![bc_application_domain]

    //![bc_application_computation]
    computation=gt::make_computation<BACKEND>(
        bc_application_domain,
        bc_preparation_coords,
        make_mss(
            execute<forward>(),
            gridtools::make_esf<functors::hexahedron_matrix_distribute<dof_per_dim,dof_per_dim,dof_per_dim> >(p_stiffness_bc()),
            gridtools::make_esf<functors::hexahedron_vector_distribute<dof_per_dim,dof_per_dim,dof_per_dim> >(p_source_bc())
        ));

    computation->ready();
    computation->steady();
    computation->run();
    computation->finalize();
    //![bc_application_computation]


//    std::cout<<"AAA"<<std::endl;
//    for (uint_t i=0; i<comp_d1; i++)
//        for (uint_t j=0; j<comp_d2; j++)
//            for (uint_t k=0; k<comp_d3; k++)
//                for(uint_t dof1 = 0;dof1<dof_per_el;++dof1)
//                    std::cout<<"element "<<i<<" "<<j<<" "<<k<<" dofs "<<dof1<<" vec mask "<<dirichlet_bc_mask_vector(i,j,k,dof1)<<std::endl;
//    std::cout<<"AAA"<<std::endl;

//    std::cout<<"BBB"<<std::endl;
//    for (uint_t i=0; i<comp_d1; i++)
//        for (uint_t j=0; j<comp_d2; j++)
//            for (uint_t k=0; k<comp_d3; k++)
//                for(uint_t dof1 = 0;dof1<dof_per_el;++dof1)
//                    for(uint_t dof2 = 0;dof2<dof_per_el;++dof2)
//                        std::cout<<"element "<<i<<" "<<j<<" "<<k<<" dofs "<<dof1<<" "<<dof2<<" matrix mask "<<dirichlet_bc_mask_matrix(i,j,k,dof1,dof2)<<std::endl;
//    std::cout<<"BBB"<<std::endl;



    //![boundary_condition_application]


//    std::cout<<"Sol calc"<<std::endl;
//
//
    std::cout<<"CCC"<<std::endl;
    std::cout<<std::setprecision(16);
    for (uint_t i=0; i<comp_d1; i++)
        for (uint_t j=0; j<comp_d2; j++)
            for (uint_t k=0; k<comp_d3; k++)
                for(uint_t dof1 = 0;dof1<dof_per_el;++dof1)
                    for(uint_t dof2 = 0;dof2<dof_per_el;++dof2)
                        std::cout<<"element "<<i<<" "<<j<<" "<<k<<" dofs "<<dof1<<" "<<dof2<<" stiff "<<stiffness(i,j,k,dof1,dof2)<<std::endl;
    std::cout<<"CCC"<<std::endl;

    std::cout<<"DDD"<<std::endl;
    for (uint_t i=0; i<comp_d1; i++)
        for (uint_t j=0; j<comp_d2; j++)
            for (uint_t k=0; k<comp_d3; k++)
                for(uint_t dof1 = 0;dof1<dof_per_el;++dof1)
                    std::cout<<"element "<<i<<" "<<j<<" "<<k<<" dofs "<<dof1<<" source "<<source(i,j,k,dof1)<<std::endl;
    std::cout<<"DDD"<<std::endl;


    //
//    std::cout<<std::endl;
//    for(uint_t dof1 = 0;dof1<dof_per_el-1;++dof1){
//
//            std::cout<<source(1,1,1,dof1)<<", \\"<<std::endl;
//    }
//    std::cout<<source(1,1,1,dof_per_el-1)<<" \\"<<std::endl;
//    std::cout<<std::endl;


    //![solve_linear_system]
    linear_solver< gdl::cg_solver<dof_per_dim, dof_per_dim, dof_per_dim> >::solve(stiffness, source, sol, stability_thr, error_thr, max_iter);
    //![solve_linear_system]

    //![print_result]
    // Print non-zero results
    for(uint_t dof = 0;dof<n_dof;++dof) {

//        if(sol.get_value(dof))
            std::cout<<dof<<" "<<sol.get_value(dof)<<std::endl;


    }
    //![print_result]

#if PRINT_DATA_FILES
    //![print_result]
    std::ofstream A_matrix;
    A_matrix.open("A_matrix.dat");
    A_matrix<<std::setprecision(16);
    for(uint_t dof1 = 0;dof1<n_dof;++dof1){
         for(uint_t dof2 = 0;dof2<n_dof;++dof2)
             A_matrix<<stiffness.get_value(dof1,dof2)<<" ";
    }
    A_matrix.close();

    std::ofstream b_vector;
    b_vector.open("b_vector.dat");
    b_vector<<std::setprecision(16);
    for(uint_t dof1 = 0;dof1<n_dof;++dof1){
            b_vector<<source.get_value(dof1)<<" ";
    }
    b_vector.close();

    std::ofstream x_vector;
    x_vector.open("x_vector.dat");
    x_vector<<std::setprecision(16);
    for(uint_t dof1 = 0;dof1<n_dof;++dof1){
        x_vector<<sol.get_value(dof1)<<" ";
    }
    x_vector.close();

    //![print_result]
#endif



}
