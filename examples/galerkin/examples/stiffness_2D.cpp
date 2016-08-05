/**
\file
*/
#define PEDANTIC_DISABLED
//! [assembly]
#include "../numerics/assembly.hpp"
//! [assembly]
#include "test_assembly.hpp"
#include "../functors/stiffness.hpp"


// [boundary integration]
int main(){

	//![definitions]
    //dimensions of the problem (in number of elements per dimension)
    auto d1=1;
    auto d2=2;
    auto d3=1;
    const auto num_dofs = 4;
    //![definitions]

    //![definitions]
    //defining the assembler, based on the Intrepid definitions for the numerics
    using matrix_storage_info_t=storage_info<  __COUNTER__, layout_tt<5> >;
    using global_stiffness_matrix_storage_info_t=storage_info<  __COUNTER__, layout_tt<5> >;
    using matrix_type=storage_t< matrix_storage_info_t >;
    using global_stiffness_matrix_type=storage_t< global_stiffness_matrix_storage_info_t >;
    using fe=reference_element<1, Lagrange, Tri>;
    using geo_map=reference_element<1, Lagrange, Tri>;
    using cub=cubature<fe::order+1, fe::shape>;
    using geo_t = intrepid::unstructured_geometry<geo_map, cub>;
    using discr_t = intrepid::discretization<fe, cub>;
    //![definitions]

    //![instantiation]
    geo_t geo_;
    discr_t fe_;
    fe_.compute(Intrepid::OPERATOR_GRAD);
    //![instantiation]

    using as=assembly< geo_t >;
    using as_base=assembly_base<geo_t>;


    //![as_instantiation]
    //constructing the integration tools
    as assembler( geo_, d1, d2, d3);
    as_base assembler_base(d1,d2,d3);
    //![as_instantiation]

    using domain_tuple_t = aggregator_type_tuple< as, as_base>;
    domain_tuple_t domain_tuple_ (assembler, assembler_base);

    //![grid]
    // First triangle
    assembler_base.grid()( 0,  0,  0,  0,  0)= 1.5;
    assembler_base.grid()( 0,  0,  0,  0,  1)= 0.;
    assembler_base.grid()( 0,  0,  0,  0,  2)= 0.;
    assembler_base.grid_map()( 0,  0,  0,  0)= 0;//Global DOF

    assembler_base.grid()( 0,  0,  0,  1,  0)= 2.;
    assembler_base.grid()( 0,  0,  0,  1,  1)= -1.;
    assembler_base.grid()( 0,  0,  0,  1,  2)= 0.;
    assembler_base.grid_map()( 0,  0,  0,  1)= 1;//Global DOF

    assembler_base.grid()( 0,  0,  0,  2,  0)= 2.;
    assembler_base.grid()( 0,  0,  0,  2,  1)= 1.;
    assembler_base.grid()( 0,  0,  0,  2,  2)= 0.;
    assembler_base.grid_map()( 0,  0,  0,  2)= 2;//Global DOF

    // Second triangle
    assembler_base.grid()( 0,  1,  0,  0,  0)= 2.;
    assembler_base.grid()( 0,  1,  0,  0,  1)= -1.;
    assembler_base.grid()( 0,  1,  0,  0,  2)= 0.;
    assembler_base.grid_map()( 0,  1,  0,  0)= 1;//Global DOF

    assembler_base.grid()( 0,  1,  0,  1,  0)= 2.5;
    assembler_base.grid()( 0,  1,  0,  1,  1)= 0.;
    assembler_base.grid()( 0,  1,  0,  1,  2)= 0.;
    assembler_base.grid_map()( 0,  1,  0,  1)= 3;//Global DOF

    assembler_base.grid()( 0,  1,  0,  2,  0)= 2.;
    assembler_base.grid()( 0,  1,  0,  2,  1)= 1.;
    assembler_base.grid()( 0,  1,  0,  2,  2)= 0.;
    assembler_base.grid_map()( 0,  1,  0,  2)= 2;//Global DOF
    //![grid]

    //![instantiation_stiffness]
    //defining the stiffness matrix: d1xd2xd3 elements
    matrix_storage_info_t meta_(d1,d2,d3,fe::basisCardinality,fe::basisCardinality);
    matrix_type stiffness_(meta_, 0.);
    //![instantiation_stiffness]

    //![instantiation_global_mass]
    global_stiffness_matrix_storage_info_t meta_global_stiffness_(num_dofs,num_dofs,1,1,1);
    global_stiffness_matrix_type global_stiffness_gt_(meta_global_stiffness_,0.,"global_stiffness_gt");
    //![instantiation_global_mass]


    /** defining the computation, i.e. for all elements:
        - computing the jacobian
        - computing its determinant
        - computing the jacobian inverse
        - integrate the stiffness matrix
        - adding the fluxes contribution
    */

    using dt = domain_tuple_t;
    //![placeholders]
    // defining the placeholder for the local gradient of the element boundary face
    typedef arg<dt::size, discr_t::grad_storage_t> p_dphi;
    // // defining the placeholder for the local values on the face
    // typedef arg<as::size+4, bd_discr_t::phi_storage_t> p_bd_phi;
    // //output
    typedef arg<dt::size+1, as_base::grid_map_type> p_grid_map;
    typedef arg<dt::size+2, matrix_type> p_stiffness;
    typedef arg<dt::size+3, global_stiffness_matrix_type> p_global_stiffness_gt;

    // appending the placeholders to the list of placeholders already in place
    auto domain=domain_tuple_.template domain<p_dphi, p_grid_map, p_stiffness, p_global_stiffness_gt>(fe_.grad(), assembler_base.grid_map(), stiffness_, global_stiffness_gt_);
    //![placeholders]


    // , m_domain(boost::fusion::make_vector(&m_grid, &m_jac, &m_fe_backend.cub_weights(), &m_jac_det, &m_jac_inv, &m_fe_backend.local_gradient(), &m_fe_bac
                                                                                                   // , &m_stiffness, &m_assembled_stiffness
    auto coords=grid<axis>({0, 0, 0, d1-1, d1},
    							  {0, 0, 0, d2-1, d2});


    coords.value_list[0] = 0;
    coords.value_list[1] = d3-1;

    //![computation]
    auto computation=make_computation<gridtools::BACKEND>(
        make_multistage
        (
            execute<forward>(),
            make_stage<functors::update_jac<geo_t> >( dt::p_grid_points(), p_dphi(), dt::p_jac())
            , make_stage<functors::det<geo_t> >(dt::p_jac(), dt::p_jac_det())
            , make_stage<functors::inv<geo_t> >(dt::p_jac(), dt::p_jac_det(), dt::p_jac_inv())
            , make_stage<functors::stiffness<fe, cub> >(dt::p_jac_det(), dt::p_jac_inv(), dt::p_weights(), p_stiffness(), p_dphi(), p_dphi())//stiffness
            ), domain, coords);

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


    auto assembly_computation=make_computation<gridtools::BACKEND>(make_multistage(execute<forward>(),
    															   make_stage<functors::global_assemble_no_if>(p_stiffness(),p_grid_map(),p_global_stiffness_gt())),
														  domain,
														  assembly_coords);
    assembly_computation->ready();
    assembly_computation->steady();
    assembly_computation->run();
    assembly_computation->finalize();
    //![assembly_computation]


    return test(assembler_base, assembler, fe_, stiffness_)==true;
}
