/**
\file
*/
#define PEDANTIC_DISABLED
#include "../numerics/assembly.hpp"
#include "../functors/mass.hpp"
#include "../numerics/assemble_storage.hpp"

constexpr float_t dx{0.1};
constexpr float_t dy{0.1};
constexpr float_t dz{0.1};


int main(){

    using namespace gridtools;
    using namespace gdl;
    using namespace gdl::enumtype;

    //![definitions]
    //defining the assembler, based on the Intrepid definitions for the numerics
    using matrix_storage_info_t=storage_info<  __COUNTER__, layout_tt<3,4> >;
    using global_mass_matrix_storage_info_t=storage_info<  __COUNTER__, layout_tt<3,4> >;
    using matrix_shrinking_storage_info_t=storage_info<  __COUNTER__, layout_tt<3,4> >;
    using matrix_type=storage_t< matrix_storage_info_t >;
    using global_mass_matrix_type=storage_t< global_mass_matrix_storage_info_t >;
    using matrix_shrinking_type=storage_t< matrix_shrinking_storage_info_t >;
    using fe=reference_element<2, Lagrange, Hexa>;
    using geo_map=reference_element<2, Lagrange, Hexa>;
    using cub=cubature<fe::order+1, fe::shape>;
    using geo_t = intrepid::unstructured_geometry<geo_map, cub>;
    using discr_t = intrepid::discretization<fe, cub>;
    using as=assembly<geo_t>;
    using as_base=assembly_base<geo_t>;
    using assemble_storage_info_t=storage_info< __COUNTER__, layout_tt<3,4> >;
    constexpr uint_t dof_per_dir{3};
    using matrix_in_storage_info_t=storage_info< __COUNTER__, layout_tt<3,4> >;
    //using matrix_in_type=storage_t< matrix_in_storage_info_t >;
    using assemble_storage_type=storage<assemble_storage< assemble_storage_info_t, dof_per_dir, dof_per_dir, dof_per_dir> >;
    //![definitions]


    assert(dof_per_dir*dof_per_dir*dof_per_dir == fe::basisCardinality);


    //![definitions]
    //dimensions of the problem (in number of elements per dimension)
    constexpr uint_t d1=3;
    constexpr uint_t d2=3;
    constexpr uint_t d3=3;
    const uint_t d4=dof_per_dir*dof_per_dir*dof_per_dir;
    const uint_t num_dofs = (d1*(dof_per_dir-1)+1)*(d2*(dof_per_dir-1)+1)*(d3*(dof_per_dir-1)+1);
    const gridtools::meta_storage_base<__COUNTER__,gridtools::layout_map<2,1,0>,false> indexing_global{(dof_per_dir-1)*d1+1, (dof_per_dir-1)*d2+1, (dof_per_dir-1)*d3+1};
    //![definitions]

    //![instantiation]
    geo_t geo_;
    discr_t fe_;
    fe_.compute(Intrepid::OPERATOR_VALUE);
    geo_.compute(Intrepid::OPERATOR_VALUE);
    geo_.compute(Intrepid::OPERATOR_GRAD);
    //![instantiation]

    //![as_instantiation]
    //constructing the integration tools
    as assembler(geo_,d1,d2,d3);
    as_base assembler_base(d1,d2,d3);
    assemble_storage_info_t ass_storage_in_(d1,d2,d3,d4,d4);
    //![as_instantiation]

    using domain_tuple_t = domain_type_tuple< as, as_base>;
    domain_tuple_t domain_tuple_ (assembler, assembler_base);


//    //![grid]
//    // First tetrahedron
//    assembler_base.grid()( 0,  0,  0,  0,  0)= 0.;
//    assembler_base.grid()( 0,  0,  0,  0,  1)= 0.;
//    assembler_base.grid()( 0,  0,  0,  0,  2)= 0.;
//    assembler_base.grid_map()( 0,  0,  0,  0)= 0;//Global DOF
//
//    assembler_base.grid()( 0,  0,  0,  1,  0)= 0.;
//    assembler_base.grid()( 0,  0,  0,  1,  1)= 0.;
//    assembler_base.grid()( 0,  0,  0,  1,  2)= 1.;
//    assembler_base.grid_map()( 0,  0,  0,  1)= 1;//Global DOF
//
//    assembler_base.grid()( 0,  0,  0,  2,  0)= 1.;
//    assembler_base.grid()( 0,  0,  0,  2,  1)= 0.;
//    assembler_base.grid()( 0,  0,  0,  2,  2)= 0.;
//    assembler_base.grid_map()( 0,  0,  0,  2)= 2;//Global DOF
//
//    assembler_base.grid()( 0,  0,  0,  3,  0)= 0.;
//    assembler_base.grid()( 0,  0,  0,  3,  1)= 1.;
//    assembler_base.grid()( 0,  0,  0,  3,  2)= 0.;
//    assembler_base.grid_map()( 0,  0,  0,  3)= 3;//Global DOF
//
//    // Second tetrahedron
//    assembler_base.grid()( 0,  1,  0,  0,  0)= 2.;
//    assembler_base.grid()( 0,  1,  0,  0,  1)= 2.;
//    assembler_base.grid()( 0,  1,  0,  0,  2)= 2.;
//    assembler_base.grid_map()( 0,  1,  0,  0)= 4;//Global DOF
//
//    assembler_base.grid()( 0,  1,  0,  1,  0)= 1.;
//    assembler_base.grid()( 0,  1,  0,  1,  1)= 0.;
//    assembler_base.grid()( 0,  1,  0,  1,  2)= 0.;
//    assembler_base.grid_map()( 0,  1,  0,  1)= 2;//Global DOF
//
//    assembler_base.grid()( 0,  1,  0,  2,  0)= 0.;
//    assembler_base.grid()( 0,  1,  0,  2,  1)= 0.;
//    assembler_base.grid()( 0,  1,  0,  2,  2)= 1.;
//    assembler_base.grid_map()( 0,  1,  0,  2)= 1;//Global DOF
//
//    assembler_base.grid()( 0,  1,  0,  3,  0)= 0.;
//    assembler_base.grid()( 0,  1,  0,  3,  1)= 1.;
//    assembler_base.grid()( 0,  1,  0,  3,  2)= 0.;
//    assembler_base.grid_map()( 0,  1,  0,  3)= 3;//Global DOF
//    //![grid]

    std::cout<<"0"<<std::endl;


    //![grid]
    //constructing a structured cartesian grid
    for (uint_t i=0; i<d1; i++)
        for (uint_t j=0; j<d2; j++)
            for (uint_t k=0; k<d3; k++){
                uint_t point = 0;
                for(uint_t dofx = 0;dofx<dof_per_dir;++dofx)
                    for(uint_t dofy = 0;dofy<dof_per_dir;++dofy)
                        for(uint_t dofz = 0;dofz<dof_per_dir;++dofz,++point){
                            assembler_base.grid()( i,  j,  k,  point,  0)= i*(dof_per_dir-1)*dx + dofx*dx;
                            assembler_base.grid()( i,  j,  k,  point,  1)= j*(dof_per_dir-1)*dy + dofy*dy;
                            assembler_base.grid()( i,  j,  k,  point,  2)= k*(dof_per_dir-1)*dz + dofz*dz;
                            assembler_base.grid_map()(i, j, k, point)=
                                    i*(dof_per_dir-1) + dofx +
                                    j*((dof_per_dir-1)*d1 + 1)*(dof_per_dir-1) + dofy*((dof_per_dir-1)*d1 + 1) +
                                    k*((dof_per_dir-1)*d1 + 1)*((dof_per_dir-1)*d2 + 1)*(dof_per_dir-1) + dofz*((dof_per_dir-1)*d1 + 1)*((dof_per_dir-1)*d2 + 1);
                        }
           }
    //![grid]


    std::cout<<"A"<<std::endl;

    //![instantiation_mass]
    //defining the mass matrix: d1xd2xd3 elements
    matrix_storage_info_t meta_(d1,d2,d3,fe::basisCardinality,fe::basisCardinality);
    matrix_type mass_(meta_, 0., "mass");
    //![instantiation_mass]

    //![instantiation_global_mass]
    global_mass_matrix_storage_info_t meta_global_mass_(num_dofs,num_dofs,1,1,1);
    global_mass_matrix_type global_mass_gt_(meta_global_mass_,0.,"global_mass_gt");
    assemble_storage_type ass_storage_in(ass_storage_in_,0,"ass_in");
    //![instantiation_global_mass]

    //![instantiation_mass]
    //defining the mass matrix: d1xd2xd3 elements
    uint_t shrinked_size = static_cast<uint_t>(std::ceil(std::sqrt(fe::basisCardinality*fe::basisCardinality - 3*(dof_per_dir*dof_per_dir*dof_per_dir*dof_per_dir) + 5)));
    std::cout<<"Original size "<<fe::basisCardinality<<" Shrinked size "<<shrinked_size<<std::endl;
    matrix_shrinking_storage_info_t meta_shrinking_(d1,d2,d3,shrinked_size,shrinked_size);
    matrix_shrinking_type mass_shrinking_(meta_shrinking_, 0., "mass_shrinking");
    //![instantiation_mass]

    //![placeholders]
    // defining the placeholder for the local basis/test functions
    using dt = domain_tuple_t;
    typedef arg<dt::size, discr_t::basis_function_storage_t> p_phi;
    typedef arg<dt::size+1, discr_t::grad_storage_t> p_dphi;
    // // defining the placeholder for the mass/global matrix values
    typedef arg<dt::size+2, as_base::grid_map_type> p_grid_map;
    typedef arg<dt::size+3, matrix_type> p_mass;
    typedef arg<dt::size+4, global_mass_matrix_type> p_global_mass_gt;
    typedef arg<dt::size+5, assemble_storage_type> p_assemble_inout;
    typedef arg<dt::size+6, matrix_shrinking_type> p_assemble_shrink;

    // appending the placeholders to the list of placeholders already in place
    auto domain=domain_tuple_.template domain<p_phi, p_dphi, p_grid_map, p_mass, p_global_mass_gt, p_assemble_inout, p_assemble_shrink>(fe_.val(), geo_.grad(), assembler_base.grid_map(), mass_, global_mass_gt_, ass_storage_in,mass_shrinking_);
    //![placeholders]


    //![placeholders]


    auto coords=grid<axis>({1, 0, 1, d1-1, d1},
                           {1, 0, 1, d2-1, d2});
    coords.value_list[0] = 1;
    coords.value_list[1] = d3-1;

    //![computation]
    auto computation=make_computation<BACKEND>(make_mss(execute<forward>(),
                                                        make_esf<functors::update_jac<geo_t> >(dt::p_grid_points(), p_dphi(), dt::p_jac()),
                                                        make_esf<functors::det< geo_t > >(dt::p_jac(), dt::p_jac_det()),
                                                        make_esf<functors::mass<fe, cub> >(dt::p_jac_det(), dt::p_weights(), p_phi(), p_phi(), p_mass())),
                                                        domain,
                                                        coords);
    computation->ready();
    computation->steady();
    computation->run();
    computation->finalize();

    std::cout<<"B"<<std::endl;


    auto computation_ass=make_computation<BACKEND>(make_mss(execute<forward>(),
                                                            make_esf<functors::update_jac<geo_t> >(dt::p_grid_points(), p_dphi(), dt::p_jac()),
                                                            make_esf<functors::det< geo_t > >(dt::p_jac(), dt::p_jac_det()),
                                                            make_esf<functors::mass<fe, cub> >(dt::p_jac_det(), dt::p_weights(), p_phi(), p_phi(), p_assemble_inout())),
                                                            domain,
                                                            coords);
    computation_ass->ready();
    computation_ass->steady();
    computation_ass->run();
    computation_ass->finalize();

//    for (uint_t i=0; i<d1; i++)
//        for (uint_t j=0; j<d2; j++)
//            for (uint_t k=0; k<d3; k++)
//                for(uint_t dof1 = 0;dof1<d4;++dof1)
//                    for(uint_t dof2 = 0;dof2<d4;++dof2){
//
////                        if(mass_(i,j,k,dof1,dof2) != ass_storage_in(i,j,k,dof1,dof2)){
//
//                        std::cout<<i<<" "<<j<<" "<<k<<" "<<dof1<<" "<<dof2<<" "<<mass_(i,j,k,dof1,dof2)<<" "<<ass_storage_in(i,j,k,dof1,dof2)<<std::endl;
////                        }
//                    }
//
//    return 0;

    //![computation]
    std::cout<<"C"<<std::endl;


    //![assembly_computation]
    auto assembly_coords=grid<axis>({0, 0, 0, num_dofs-1, num_dofs},
                                    {0, 0, 0, num_dofs-1, num_dofs});
    assembly_coords.value_list[0] = 0;
    assembly_coords.value_list[1] = 0;


    auto assembly_computation=make_computation<BACKEND>(make_mss(execute<forward>(),
                                                                 make_esf<functors::global_assemble_no_if>(p_mass(),p_grid_map(),p_global_mass_gt())),
                                                                 domain,
                                                                 assembly_coords);
    assembly_computation->ready();
    assembly_computation->steady();
    assembly_computation->run();
    assembly_computation->finalize();

    std::cout<<"D"<<std::endl;


    auto computation_ass_new= make_computation<BACKEND >(
            make_mss( execute<forward>(),
                      make_esf< functors::hexahedron_assemble<dof_per_dir,dof_per_dir,dof_per_dir> >(p_assemble_inout(),p_assemble_inout())),
                      domain, coords);

    computation_ass_new->ready();
    computation_ass_new->steady();
    computation_ass_new->run();
    computation_ass_new->finalize();

    std::cout<<"E"<<std::endl;


//    ushort_t I1;
//    ushort_t J1;
//    ushort_t K1;
//    ushort_t i1;
//    ushort_t j1;
//    ushort_t k1;
//    ushort_t I2;
//    ushort_t J2;
//    ushort_t K2;
//    ushort_t i2;
//    ushort_t j2;
//    ushort_t k2;

//    for(ushort_t Id1=1; Id1<indexing_global.template dims<0>()-1; Id1++)
//    for(ushort_t Jd1=1; Jd1<indexing_global.template dims<1>()-1; Jd1++)
//    for(ushort_t Kd1=1; Kd1<indexing_global.template dims<2>()-1; Kd1++)
//        for(ushort_t Id2=1; Id2<indexing_global.template dims<0>()-1; Id2++)
//            for(ushort_t Jd2=1; Jd2<indexing_global.template dims<1>()-1; Jd2++)
//                for(ushort_t Kd2=1; Kd2<indexing_global.template dims<2>()-1; Kd2++){
//                        //                                    std::cout<<Id1<<" "<<Jd1<<" "<<Kd1<<" "
//                        //                                                    <<Id2<<" "<<Jd2<<" "<<Kd2<<std::endl;
//                    uint_t global_dof1=indexing_global.index(Id1,Jd1,Kd1);
//                    uint_t global_dof2=indexing_global.index(Id2,Jd2,Kd2);
//
//
//    //                                if(findLocalDofIndexes(Id1,Jd1,Kd1,Id2,Jd2,Kd2,
//    //                                                       I1,J1,K1,i1,j1,k1,
//    //                                                       I2,J2,K2,i2,j2,k2)==false){
//    //                                        //                                                            std::cout<<"False"<<std::endl;
//    //
//    //                                    //                                                            std::cout<<I1<<" "<<J1<<" "<<K1<<" "
//    //                                    //                                                                                     <<I2<<" "<<J2<<" "<<K2<<" "
//    //                                    //                                                                                     <<i1<<" "<<j1<<" "<<k1<<" "
//    //                                    //                                                                                     <<i2<<" "<<j2<<" "<<k2<<" "
//    //                                    //                                                                                     <<indexing_local.index(k1,j1,i1)<<" "
//    //                                    //                                                                                     <<indexing_local.index(k2,j2,i2)<<std::endl;
//    //
//    //
//    //                                    if(in(I1,J1,K1,indexing_local.index(k1,j1,i1)*d4 + indexing_local.index(k2,j2,i2)))
//    //                                    {
//    //                                      std::cout<<"Value"<<std::endl;
//    //                                      std::cout<<in(I1,J1,K1,indexing_local.index(k1,j1,i1)*d4 + indexing_local.index(k2,j2,i2))<<" "<<out_test(global_dof1,global_dof2,0,0,0)<<std::endl;
//    //                              }
//    //                                                                                          }
//    //                                                                                          else
//    //                                                                                          {
//    //                                                                                                  std::cout<<"True"<<std::endl;
//    //                                    ////                                                          std::cout<<I1<<" "<<J1<<" "<<K1<<" "
//    //                                    ////                                                                           <<I2<<" "<<J2<<" "<<K2<<" "
//    //                                    ////                                                                           <<i1<<" "<<j1<<" "<<k1<<" "
//    //                                    ////                                                                           <<i2<<" "<<j2<<" "<<k2<<std::endl;
//    //                                    ////                                  std::cout<<0<<" "<<out_test(global_dof1,global_dof2,0,0,0)<<std::endl;
//    //                                                                                          }
//    //                                                                                          std::cout<<std::endl;
//    //
//    //                                    double in_val=0;
//    //                                    if(findLocalDofIndexes(Id1,Jd1,Kd1,Id2,Jd2,Kd2,
//    //                                                           I1,J1,K1,i1,j1,k1,
//    //                                                           I2,J2,K2,i2,j2,k2)==false)
//    //                                        {
//    //                                            in_val = in(I1,J1,K1,indexing_local.index(i1,j1,k1)*d4 + indexing_local.index(i2,j2,k2));
//    //                                        }
//    //
//    //                                    if(in_val!=0)
//    //                                    {
//    //                                            std::cout<<Id1<<" "<<Jd1<<" "<<Kd1<<" "<<Id2<<" "<<Jd2<<" "<<Kd2<<std::endl;
//    //                                            std::cout<<I1<<" "<<J1<<" "<<K1<<" "<<I2<<" "<<J2<<" "<<K2<<" "<<i1<<" "<<j1<<" "<<k1<<" "<<i2<<" "<<j2<<" "<<k2<<" "<<in_val<<std::endl;
//    //                                            std::cout<<std::endl;
//    //                                    }
//    //
//    //                                    if(in_val!=ass_storage_in.get_value(Id1,Jd1,Kd1,Id2,Jd2,Kd2))
//    //                                    {
//    //                                            std::cout<<Id1<<" "<<Jd1<<" "<<Kd1<<" "<<Id2<<" "<<Jd2<<" "<<Kd2<<std::endl;
//    //                                            std::cout<<I1<<" "<<J1<<" "<<K1<<" "<<I2<<" "<<J2<<" "<<K2<<" "<<i1<<" "<<j1<<" "<<k1<<" "<<i2<<" "<<j2<<" "<<k2<<" "<<in_val<<std::endl;
//    //                                            std::cout<<in_val<<" "<< ass_storage_in.get_value(Id1,Jd1,Kd1,Id2,Jd2,Kd2)<<std::endl;
//    //                                            std::cout<<std::endl;
//    //                                    }
//    //
//    //
//    //                                                                                          std::cout<<in_val<<" "<<out_test(global_dof1,global_dof2,0,0,0)<<std::endl;
//    //                                    if(in_val != out_test(global_dof1,global_dof2,0,0,0))
//    //                                        {
//    //                                            std::cout<<"Failed"<<std::endl;
//    //                                            std::cout<<in_val<<" "<<out_test(global_dof1,global_dof2,0,0,0)<<std::endl;
//    //                                            std::cout<<Id1<<" "<<Jd1<<" "<<Kd1<<" "<<Id2<<" "<<Jd2<<" "<<Kd2<<std::endl;
//    //                                            std::cout<<global_dof1<<" "<<global_dof2<<std::endl;
//    //                                            std::cout<<I1<<" "<<J1<<" "<<K1<<" "<<I2<<" "<<J2<<" "<<K2<<" "<<i1<<" "<<j1<<" "<<k1<<" "<<i2<<" "<<j2<<" "<<k2<<" "<<indexing_local.index(i1,j1,k1)<<" "<<indexing_local.index(i2,j2,k2)<<std::endl;
//    //                                            std::cout<<std::endl;
//    //                                            return 1;
//    //                                        }
//
//
//                        if(std::abs(global_mass_gt_(global_dof1,global_dof2,0,0,0)-ass_storage_in.get_value(Id1,Jd1,Kd1,Id2,Jd2,Kd2))>1.e-17)
//                        {
//                            std::cout<<global_dof1<<" "<<global_dof2<<std::endl;
//
//                            std::cout<<Id1<<" "<<Jd1<<" "<<Kd1<<" "<<Id2<<" "<<Jd2<<" "<<Kd2<<std::endl;
//                            std::cout<<global_mass_gt_(global_dof1,global_dof2,0,0,0)<<" "<< ass_storage_in.get_value(Id1,Jd1,Kd1,Id2,Jd2,Kd2)<<std::endl;
//                            std::cout<<std::endl;
//                        }
//
//
//
//                    }


//    auto computation_shrink= make_computation<BACKEND >(
//            make_mss( execute<forward>(),
//                      make_esf< functors::hexahedron_assemble_shrinking<dof_per_dir,dof_per_dir,dof_per_dir> >(p_assemble_inout(),p_assemble_shrink())),
//                      domain, coords);
//
//    computation_shrink->ready();
//    computation_shrink->steady();
//    computation_shrink->run();
//    computation_shrink->finalize();


   // [assembly_computation]

    return 0;

}
