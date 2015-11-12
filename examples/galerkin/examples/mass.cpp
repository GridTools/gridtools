/**
\file
*/
#pragma once
#define PEDANTIC_DISABLED
#include "../functors/assembly.hpp"
#include "test_assembly.hpp"

// [integration]
template <typename FE, typename Cubature>
struct integration {
    using jac_det =accessor<0, range<0,0,0,0> , 4> const;
    using weights =accessor<1, range<0,0,0,0> , 3> const;
    using mass    =accessor<2, range<0,0,0,0> , 5> ;
    using phi    =accessor<3, range<0,0,0,0> , 3> const;
    using psi    =accessor<4, range<0,0,0,0> , 3> const;
    using arg_list= boost::mpl::vector<jac_det, weights, mass, phi, psi> ;
    using quad=dimension<4>;

    using fe=FE;
    using cub=Cubature;

    template <typename Evaluation>
    GT_FUNCTION
    static void Do(Evaluation const & eval, x_interval) {

        quad::Index qp;
        dimension<5>::Index dimx;
        dimension<6>::Index dimy;
        // static int_t dd=fe::hypercube_t::boundary_w_codim<2>::n_points::value;

        //projection of f on a (e.g.) P1 FE space ReferenceFESpace1:
        //loop on quadrature nodes, and on nodes of the P1 element (i,j,k) with i,j,k\in {0,1}
        for(short_t P_i=0; P_i<fe::basisCardinality; ++P_i) // current dof
        {
            for(short_t Q_i=0; Q_i<fe::basisCardinality; ++Q_i)
            {//other dofs whose basis function has nonzero support on the element
            	for(short_t q=0; q<cub::numCubPoints(); ++q){
                    eval(mass(0,0,0,P_i,Q_i))  +=
                        eval(!phi(P_i,q,0)*(!psi(Q_i,q,0))*jac_det(qp+q)*!weights(q,0,0));
                }
            }
        }
    }
};
// [integration]

int main(){
	//![definitions]
    //dimensions of the problem (in number of elements per dimension)
    auto d1=8;
    auto d2=8;
    auto d3=1;
    //![definitions]
    //defining the assembler, based on the Intrepid definitions for the numerics
	using matrix_storage_info_t=storage_info< layout_tt<3,4> , __COUNTER__>;
    using matrix_type=storage_t< matrix_storage_info_t >;
    using fe=reference_element<1, Lagrange, Hexa>;
    using geo_map=reference_element<1, Lagrange, Hexa>;
    using cub=cubature<fe::order, fe::shape>;
    using geo_t = intrepid::geometry<geo_map, cub>;
    using discr_t = intrepid::discretization<fe, cub>;

    //![instantiation]
    geo_t geo_;
    discr_t fe_;
    fe_.compute(Intrepid::OPERATOR_VALUE);
    fe_.compute(Intrepid::OPERATOR_GRAD);
    //![instantiation]

    using as=assembly<geo_t>;
    using as_base=assembly_base<geo_t>;

    //![as_instantiation]
    //constructing the integration tools
    as assembler(geo_,d1,d2,d3);
    as assembler_grad(geo_,d1,d2,d3);
    as_base assembler_base(d1,d2,d3);
    //![as_instantiation]

    using domain_tuple_t = domain_type_tuple< as, as, as_base>;
    domain_tuple_t domain_tuple_ (assembler, assembler_grad, assembler_base);

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
                }
    //![grid]


    //![instantiation_mass]
    //defining the mass matrix: d1xd2xd3 elements
    matrix_storage_info_t meta_(d1,d2,d3,fe::basisCardinality,fe::basisCardinality);
    matrix_type mass_(meta_, 0., "mass");
    //![instantiation_mass]


    using dt = domain_tuple_t;

    //![placeholders]
    // defining the placeholder for the local basis/test functions
    typedef arg<dt::size, discr_t::basis_function_storage_t> p_phi;
    typedef arg<dt::size+1, discr_t::grad_storage_t> p_dphi;
    // // defining the placeholder for the mass matrix values
    typedef arg<dt::size+2, matrix_type> p_mass;


    // appending the placeholders to the list of placeholders already in place
    auto domain=domain_tuple_.template domain<p_phi, p_dphi, p_mass>(fe_.val(), fe_.grad(), mass_);
    //![placeholders]


    auto coords=coordinates<axis>({1, 0, 1, d1-1, d1},
                            	  {1, 0, 1, d2-1, d2});
    coords.value_list[0] = 0;
    coords.value_list[1] = d3-1;


    //![computation]
    auto computation=make_computation<gridtools::BACKEND>(make_mss(execute<forward>(),
								   make_esf<functors::update_jac<geo_t> >(dt::p_grid_points(), p_dphi(), dt::p_jac()),
								   make_esf<functors::det<geo_t> >(dt::p_jac(), dt::p_jac_det()),
								   make_esf<integration<fe, cub> >(dt::p_jac_det(), dt::p_weights(), p_mass(), p_phi(), p_phi())),
							  domain,
							  coords);


    computation->ready();
    computation->steady();
    computation->run();
    computation->finalize();
    //![computation]

//    return test_mass(assembler_base, assembler, fe_, mass_)==true;
}
