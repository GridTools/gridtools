/**
\file testing the gathering of the information from the neighbours
*/
#define PEDANTIC_DISABLED
#define HAVE_INTREPID_DEBUG
//! [assembly]
#include "assembly.hpp"
//! [assembly]


int main(){
    //![definitions]
    using namespace enumtype;
    using namespace gridtools;
    //defining the assembler, based on the Intrepid definitions for the numerics
    using matrix_storage_info_t=storage_info< layout_tt<3> >;
    using matrix_type=storage_t< matrix_storage_info_t >;

    using fe=reference_element<1, Lagrange, Hexa>;
    using geo_map=reference_element<1, Lagrange, Hexa>;
    using cub=cubature<fe::order, fe::shape>;
    using geo_t = intrepid::geometry<geo_map, cub>;

    geo_t discr_;

    discr_.compute(Intrepid::OPERATOR_VALUE);

    //![boundary]

    using as=assembly<discr_t, geo_t>;

    //![definitions]

    //dimensions of the problem (in number of elements per dimension)
    auto d1=8;
    auto d2=8;
    auto d3=1;

    geo_t geo;
    //![as_instantiation]
    //constructing the integration tools on the boundary
    as assembler(discr_,d1,d2,d3);
    //![as_instantiation]

    //![grid]
    //constructing a structured cartesian grid
    for (uint_t i=0; i<d1; i++)
        for (uint_t j=0; j<d2; j++)
            for (uint_t k=0; k<d3; k++)
                for (uint_t point=0; point<fe::basisCardinality; point++)
                {
                    assembler.grid()( i,  j,  k,  point,  0)= (i + geo.grid()(point, 0));
                    assembler.grid()( i,  j,  k,  point,  1)= (j + geo.grid()(point, 1));
                    assembler.grid()( i,  j,  k,  point,  2)= (k + geo.grid()(point, 2));
                }
    //![grid]

    //![instantiation_stiffness]
    //defining the stiffness matrix: d1xd2xd3 elements
    matrix_storage_info_t meta_(d1,d2,d3,fe::basisCardinality,fe::basisCardinality);
    matrix_type mass_(meta_, 1., "mass");

    //![placeholders]
    // defining the placeholder for the mass
    typedef arg<as::size, matrix_type> p_mass;
    // defining the placeholder for the local gradient of the element boundary face
    typedef arg<as::size+1, bd_discr_t::grad_storage_t> p_bd_dphi;

    typedef arg<as::size+2, bd_discr_t::basis_function_storage_t> p_bd_phi;

    // appending the placeholders to the list of placeholders already in place
    auto domain=assembler.template domain<p_mass, p_bd_dphi, p_bd_phi>(mass_, discr_.local_gradient(), discr_.phi());
    //![placeholders]


    auto coords=coordinates<axis>({1, 0, 1, d1-1, d1},
        {1, 0, 1, d2-1, d2});
    coords.value_list[0] = 0;
    coords.value_list[1] = d3-1;

    //![computation]
    auto computation=make_computation< gridtools::BACKEND >(
        make_mss
        (
            execute<forward>()(
                // subtracts the values at the element boundaries "south"
                make_esf< functors::assemble< geo_map_t, subtract_functor > >(p_mass(), p_mass(), p_mass())
                ), domain, coords);

    computation->ready();
    computation->steady();
    computation->run();
    computation->finalize();
    //![computation]
    intrepid::test(assembler, bd_discr_, mass_);
}
