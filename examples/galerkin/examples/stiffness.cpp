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
    auto d1=8;
    auto d2=8;
    auto d3=1;
    const auto num_dofs = 1;
    //![definitions]

    //![definitions]
    //defining the assembler, based on the Intrepid definitions for the numerics
	using matrix_storage_info_t=storage_info< layout_tt<3,4> , __COUNTER__>;
    using matrix_type=storage_t< matrix_storage_info_t >;
    using fe=reference_element<1, Lagrange, Hexa>;
    using geo_map=reference_element<1, Lagrange, Hexa>;
    using cub=cubature<fe::order+1, fe::shape>;
    using geo_t = intrepid::geometry<geo_map, cub>;
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
    as_base assembler_base(d1,d2,d3,num_dofs);
    //![as_instantiation]

    using domain_tuple_t = domain_type_tuple< as, as_base>;
    domain_tuple_t domain_tuple_ (assembler, assembler_base);

    //![grid]
    //constructing a structured cartesian grid
    for (uint_t i=0; i<d1; i++)
        for (uint_t j=0; j<d2; j++)
            for (uint_t k=0; k<d3; k++)
                for (uint_t point=0; point<fe::basisCardinality; point++)
                {
                    assembler_base.grid()( i,  j,  k,  point,  0)= (i + geo_.grid()(point, 0));
                    assembler_base.grid()( i,  j,  k,  point,  1)= (j + geo_.grid()(point, 1));
                    assembler_base.grid()( i,  j,  k,  point,  2)= (k + geo_.grid()(point, 2));
                    assembler_base.grid_map()(i,j,k,point)=0;//Global DOF // TODO: assign correct values
                }
    //![grid]


    //![instantiation_stiffness]
    //defining the stiffness matrix: d1xd2xd3 elements
    matrix_storage_info_t meta_(d1,d2,d3,fe::basisCardinality,fe::basisCardinality);
    matrix_type stiffness_(meta_, 0.);
    //![instantiation_stiffness]

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
    typedef arg<dt::size+1, matrix_type> p_stiffness;

    // appending the placeholders to the list of placeholders already in place
    auto domain=domain_tuple_.template domain<p_dphi, p_stiffness>(fe_.grad(), stiffness_);
    //![placeholders]


    // , m_domain(boost::fusion::make_vector(&m_grid, &m_jac, &m_fe_backend.cub_weights(), &m_jac_det, &m_jac_inv, &m_fe_backend.local_gradient(), &m_fe_bac
                                                                                                   // , &m_stiffness, &m_assembled_stiffness
    auto coords=coordinates<axis>({0, 0, 0, d1-1, d1},
    							  {0, 0, 0, d2-1, d2});
    coords.value_list[0] = 0;
    coords.value_list[1] = d3-1;

    //![computation]
    auto computation=make_computation<gridtools::BACKEND>(
        make_mss
        (
            execute<forward>(),
            make_esf<functors::update_jac<geo_t> >( dt::p_grid_points(), p_dphi(), dt::p_jac())
            , make_esf<functors::det<geo_t> >(dt::p_jac(), dt::p_jac_det())
            , make_esf<functors::inv<geo_t> >(dt::p_jac(), dt::p_jac_det(), dt::p_jac_inv())
            , make_esf<functors::stiffness<fe, cub> >(dt::p_jac_det(), dt::p_jac_inv(), dt::p_weights(), p_stiffness(), p_dphi(), p_dphi())//stiffness
            ), domain, coords);

    computation->ready();
    computation->steady();
    computation->run();
    computation->finalize();
    //![computation]

    return test(assembler_base, assembler, fe_, stiffness_)==true;
}
