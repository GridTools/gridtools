/**
\file
*/
#define PEDANTIC_DISABLED
#include "../numerics/assembly.hpp"
#include "test_assembly.hpp"
#include "../functors/mass.hpp"

using namespace gdl;
using namespace enumtype;

int main( int argc, char ** argv){

    if(argc!=4) {
        printf("usage: \n >> mass <d1> <d2> <d3>\n");
        exit(-666);
    }
	//![definitions]
    //dimensions of the problem (in number of elements per dimension)
    uint_t d1= atoi(argv[1]);
    uint_t d2= atoi(argv[2]);
    uint_t d3= atoi(argv[3]);
    //![definitions]

    //![definitions]
    //defining the assembler, based on the Intrepid definitions for the numerics
    using matrix_storage_info_t=storage_info< __COUNTER__, layout_tt<5> >;
    using matrix_type=storage_t< matrix_storage_info_t >;
    using fe=reference_element<3, Lagrange, Hexa>;
    using geo_map=reference_element<1, Lagrange, Hexa>;
    using cub=cubature<4, fe::shape()>;
    using geo_t = intrepid::geometry<geo_map, cub>;
    using discr_t = intrepid::discretization<fe, cub>;
    //![definitions]

    //![instantiation]
    geo_t geo_;
    discr_t fe_;
    fe_.compute(Intrepid::OPERATOR_VALUE);
    geo_.compute(Intrepid::OPERATOR_VALUE);
    geo_.compute(Intrepid::OPERATOR_GRAD);
    //![instantiation]

#ifdef NDEBUG
    constexpr
#endif
    gt::dimension<1> i;
    gt::dimension<2> j;
    gt::dimension<4> row;

    using as=assembly<geo_t>;
    using as_base=assembly_base<geo_t>;

    //![as_instantiation]
    //constructing the integration tools
    as assembler(geo_,d1,d2,d3);
    as_base assembler_base(d1,d2,d3);
    //![as_instantiation]

    //![grid]
    //constructing a structured cartesian grid
    for (uint_t i=0; i<d1; i++)
        for (uint_t j=0; j<d2; j++)
            for (uint_t k=0; k<d3; k++)
            {
                for (uint_t point=0; point<geo_map::basis_cardinality(); point++)
                {
                    assembler_base.grid()( i,  j,  k,  point,  0)= (i + geo_.grid()(point, 0, 0));
                    assembler_base.grid()( i,  j,  k,  point,  1)= (j + geo_.grid()(point, 1, 0));
                    assembler_base.grid()( i,  j,  k,  point,  2)= (k + geo_.grid()(point, 2, 0));
                    // assembler_base.grid_map()(i,j,k,point)=0;//Global DOF // TODO: assign correct values
                }
            }
    //![grid]


    //![instantiation_mass]
    //defining the mass matrix: d1xd2xd3 elements
    matrix_storage_info_t meta_(d1,d2,d3,fe::basis_cardinality(),fe::basis_cardinality());
    matrix_type mass_(meta_, 0., "mass");
    //![instantiation_mass]

    typedef gt::arg<0, typename as_base::grid_type >       p_grid_points;
    typedef gt::arg<1, typename as::jacobian_type >   p_jac;
    typedef gt::arg<2, typename as::geometry_t::weights_storage_t >   p_weights;
    typedef gt::arg<3, typename as::storage_type >    p_jac_det;
    typedef gt::arg<4, typename as::jacobian_type >   p_jac_inv;
    typedef gt::arg<5, typename as::geometry_t::basis_function_storage_t> p_phi_geo;
    typedef gt::arg<6, typename as::geometry_t::grad_storage_t> p_dphi_geo;

    //![placeholders]
    // defining the placeholder for the local basis/test functions
    typedef gt::arg<7, typename discr_t::basis_function_storage_t> p_phi;
    // // defining the placeholder for the mass matrix values
    typedef gt::arg<8, matrix_type> p_mass;


    typedef boost::mpl::vector<p_grid_points, p_jac, p_weights, p_jac_det, p_jac_inv, p_phi_geo, p_dphi_geo, p_phi, p_mass> arg_list;
    // appending the placeholders to the list of placeholders already in place
    gt::aggregator_type<arg_list> domain(
	boost::fusion::make_vector(   &assembler_base.grid()
				      , &assembler.jac()
				      , &assembler.fe_backend().cub_weights()
				      , &assembler.jac_det()
				      , &assembler.jac_inv()
				      , &assembler.fe_backend().val()
				      , &assembler.fe_backend().grad()
				      , &fe_.val()
				      , &mass_) );

    //![placeholders]

    auto coords=gt::grid<axis>({0, 0, 0, d1-1, d1},
                            	  {0, 0, 0, d2-1, d2});
    coords.value_list[0] = 0;
    coords.value_list[1] = d3-1;

    //![computation]
    auto computation=gt::make_computation<BACKEND>
        (         domain,
                  coords,
                  make_multistage(execute<forward>(),
                           gt::make_stage<functors::update_jac<geo_t> >(p_grid_points(), p_dphi_geo(), p_jac()),
                           gt::make_stage<functors::det< geo_t > >(p_jac(), p_jac_det()),
                           gt::make_stage<functors::mass >(p_jac_det(), p_weights(), p_phi(), p_phi(), p_mass()))
            );


    computation->ready();
    computation->steady();
    computation->run();
    computation->finalize();
    //![computation]
#ifndef CUDA_EXAMPLE
    std::cout << computation->print_meter() << std::endl;
#endif
    return test_mass(assembler_base, assembler, fe_, mass_)==true;
}
