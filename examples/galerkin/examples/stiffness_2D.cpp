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
	using matrix_storage_info_t=storage_info< layout_tt<3,4> , __COUNTER__>;
    using matrix_type=storage_t< matrix_storage_info_t >;
    using fe=reference_element<1, Lagrange, Tri>;
    using geo_map=reference_element<1, Lagrange, Tri>;
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
    matrix_storage_info_t meta_global_stiffness_(num_dofs,num_dofs,1,1,1);
    matrix_type global_stiffness_(meta_global_stiffness_,0.,"global_stiffness");
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

    //![global mass matrix assembly]
	for(unsigned int i=0;i<d1;++i)
	{
		for(unsigned int j=0;j<d2;++j)
		{
			for(unsigned int k=0;k<d3;++k)
			{
				for(auto l_dof1=0;l_dof1<fe::basisCardinality;++l_dof1)
				{
					const u_int P=assembler_base.get_grid_map()(i,j,k,l_dof1);
					for(auto l_dof2=0;l_dof2<fe::basisCardinality;++l_dof2)
					{
						const u_int Q=assembler_base.get_grid_map()(i,j,k,l_dof2);
						global_stiffness_(P,Q,0,0,0) += stiffness_(i,j,k,l_dof1,l_dof2);
					}
				}
			}
		}
	}
    //![global mass matrix assembly]


    return test(assembler_base, assembler, fe_, stiffness_)==true;
}
