/**
\file
*/
#define PEDANTIC_DISABLED
#include "../numerics/assembly.hpp"
#include "test_assembly.hpp"
#include "../functors/mass.hpp"
#include "../mesh/dolfin_mesh.hpp"

// "Manual" grid definition
//    //![definitions]
//    //dimensions of the problem (in number of elements per dimension)
//    auto d1=1;
//    auto d2=2;
//    auto d3=1;
//    const auto num_dofs = 4;
//    //![definitions]
//
//
//    //![grid]
//    // First triangle
//    assembler_base.grid()( 0,  0,  0,  0,  0)= 1.5;
//    assembler_base.grid()( 0,  0,  0,  0,  1)= 0.;
//    assembler_base.grid()( 0,  0,  0,  0,  2)= 0.;
//    assembler_base.grid_map()( 0,  0,  0,  0)= 0;//Global DOF
//
//    assembler_base.grid()( 0,  0,  0,  1,  0)= 2.;
//    assembler_base.grid()( 0,  0,  0,  1,  1)= -1.;
//    assembler_base.grid()( 0,  0,  0,  1,  2)= 0.;
//    assembler_base.grid_map()( 0,  0,  0,  1)= 1;//Global DOF
//
//    assembler_base.grid()( 0,  0,  0,  2,  0)= 2.;
//    assembler_base.grid()( 0,  0,  0,  2,  1)= 1.;
//    assembler_base.grid()( 0,  0,  0,  2,  2)= 0.;
//    assembler_base.grid_map()( 0,  0,  0,  2)= 2;//Global DOF
//
//    // Second triangle
//    assembler_base.grid()( 0,  1,  0,  0,  0)= 2.;
//    assembler_base.grid()( 0,  1,  0,  0,  1)= -1.;
//    assembler_base.grid()( 0,  1,  0,  0,  2)= 0.;
//    assembler_base.grid_map()( 0,  1,  0,  0)= 1;//Global DOF
//
//    assembler_base.grid()( 0,  1,  0,  1,  0)= 2.5;
//    assembler_base.grid()( 0,  1,  0,  1,  1)= 0.;
//    assembler_base.grid()( 0,  1,  0,  1,  2)= 0.;
//    assembler_base.grid_map()( 0,  1,  0,  1)= 3;//Global DOF
//
//    assembler_base.grid()( 0,  1,  0,  2,  0)= 2.;
//    assembler_base.grid()( 0,  1,  0,  2,  1)= 1.;
//    assembler_base.grid()( 0,  1,  0,  2,  2)= 0.;
//    assembler_base.grid_map()( 0,  1,  0,  2)= 2;//Global DOF
//    //![grid]



int main(){

    //![load_mesh]
    dolfin_mesh mesh("/users/bignamic/Development/GMESHExamples/2D_square.xml");
    std::cout<<"Mesh loading completed"<<std::endl;
    //![load_mesh]

	//![definitions]
    //dimensions of the problem (in number of elements per dimension)
    const u_int d1=1;
    const u_int d2=mesh.num_elements();
    const u_int d3=1;
    const u_int num_dofs=mesh.num_dofs();
    //![definitions]

    //![definitions]
    //defining the assembler, based on the Intrepid definitions for the numerics
    using matrix_storage_info_t=storage_info<  __COUNTER__, layout_tt<3,4> >;
    using global_mass_matrix_storage_info_t=storage_info<  __COUNTER__, layout_tt<3,4> >;
    using matrix_type=storage_t< matrix_storage_info_t >;
    using global_mass_matrix_type=storage_t< global_mass_matrix_storage_info_t >;
    using fe=reference_element<1, Lagrange, Tri>;
    using geo_map=reference_element<1, Lagrange, Tri>;
    using cub=cubature<fe::order+1, fe::shape>;
    using geo_t = intrepid::unstructured_geometry<geo_map, cub>;
    using discr_t = intrepid::discretization<fe, cub>;
    //![definitions]

    //![instantiation]
    geo_t geo_;
    discr_t fe_;
    fe_.compute(Intrepid::OPERATOR_VALUE);
    geo_.compute(Intrepid::OPERATOR_VALUE);
    geo_.compute(Intrepid::OPERATOR_GRAD);
    //![instantiation]

    using as=assembly<geo_t>;
    using as_base=assembly_base<geo_t>;

    //![as_instantiation]
    //constructing the integration tools
    as assembler(geo_,d1,d2,d3);
    as_base assembler_base(d1,d2,d3);
    //![as_instantiation]

    using domain_tuple_t = domain_type_tuple< as, as_base>;
    domain_tuple_t domain_tuple_ (assembler, assembler_base);

    //![grid]
    mesh.build_grid(assembler_base);
    //![grid]

    //![instantiation_mass]
    //defining the mass matrix: d1xd2xd3 elements
    matrix_storage_info_t meta_(d1,d2,d3,fe::basisCardinality,fe::basisCardinality);
    matrix_type mass_(meta_, 0., "mass");
    //![instantiation_mass]

    //![instantiation_global_mass]
    global_mass_matrix_storage_info_t meta_global_mass_(num_dofs,num_dofs,1,1,1);
    global_mass_matrix_type global_mass_gt_(meta_global_mass_,0.,"global_mass_gt");
    //![instantiation_global_mass]

    using dt = domain_tuple_t;

    //![placeholders]
    // defining the placeholder for the local basis/test functions
    typedef arg<dt::size, discr_t::basis_function_storage_t> p_phi;
    typedef arg<dt::size+1, discr_t::grad_storage_t> p_dphi;
    // // defining the placeholder for the mass/global matrix values
    typedef arg<dt::size+2, as_base::grid_map_type> p_grid_map;
    typedef arg<dt::size+3, matrix_type> p_mass;
    typedef arg<dt::size+4, global_mass_matrix_type> p_global_mass_gt;


    // appending the placeholders to the list of placeholders already in place
    auto domain=domain_tuple_.template domain<p_phi, p_dphi, p_grid_map, p_mass, p_global_mass_gt>(fe_.val(), geo_.grad(), assembler_base.grid_map(), mass_, global_mass_gt_);
    //![placeholders]

    auto coords=grid<axis>({0, 0, 0, d1-1, d1},
                            	  {0, 0, 0, d2-1, d2});
    coords.value_list[0] = 0;
    coords.value_list[1] = d3-1;

    //![computation]
    auto computation=make_computation<gridtools::BACKEND>(make_mss(execute<forward>(),
    															   make_esf<functors::update_jac<geo_t> >(dt::p_grid_points(), p_dphi(), dt::p_jac()),
    															   make_esf<functors::det< geo_t > >(dt::p_jac(), dt::p_jac_det()),
																   make_esf<functors::mass<fe, cub> >(dt::p_jac_det(), dt::p_weights(), p_phi(), p_phi(), p_mass())),
														  domain,
														  coords);
    computation->ready();
    computation->steady();
    computation->run();
    computation->finalize();
    //![computation]


    //![assembly_computation]
    auto assembly_coords=grid<axis>({0, 0, 0, num_dofs-1, num_dofs},
										   {0, 0, 0, num_dofs-1, num_dofs});
    assembly_coords.value_list[0] = 0;
    assembly_coords.value_list[1] = 0;


    auto assembly_computation=make_positional_computation<gridtools::BACKEND>(make_mss(execute<forward>(),
    															   make_esf<functors::global_assemble>(p_mass(),p_grid_map(),p_global_mass_gt())),
														  domain,
														  assembly_coords);
    assembly_computation->ready();
    assembly_computation->steady();
    assembly_computation->run();
    assembly_computation->finalize();
    //![assembly_computation]

    //![computation]
    return test_mass(assembler_base, assembler, fe_, mass_)==true;
}
