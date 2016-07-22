/**
   \file
*/

#define PEDANTIC_DISABLED
//! [assembly]
#include "../numerics/assembly.hpp"
//! [assembly]
#include "test_assembly.hpp"
#include "../functors/stiffness.hpp"

using namespace gdl;
using namespace enumtype;

int main( int argc, char ** argv){

    if(argc!=4) {
        printf("usage: \n >> stiffness <d1> <d2> <d3>\n");
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
#ifdef CUDA_EXAMPLE
    using matrix_storage_info_t=storage_info< __COUNTER__, layout_tt<4,3> >;
#else
    using matrix_storage_info_t=storage_info< __COUNTER__, layout_tt<3,4> >;
#endif
    using matrix_type = storage_t< matrix_storage_info_t >;
    using fe = reference_element<1, Lagrange, Hexa>;
    using geo_map = reference_element<1, Lagrange, Hexa>;
    using cub = cubature<2, fe::shape()>;
    using geo_t = intrepid::geometry<geo_map, cub>;
    using discr_t = intrepid::discretization<fe, cub>;
    //![definitions]

    //![instantiation]
    geo_t geo_;
    discr_t fe_;
    geo_.compute(Intrepid::OPERATOR_GRAD);//redundants
    geo_.compute(Intrepid::OPERATOR_VALUE);
    fe_.compute(Intrepid::OPERATOR_GRAD);
    fe_.compute(Intrepid::OPERATOR_VALUE);
    //![instantiation]

    using as=assembly< geo_t >;
    using as_base=assembly_base<geo_t>;


    //![as_instantiation]
    //constructing the integration tools
    as assembler( geo_, d1, d2, d3);
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


    //![instantiation_stiffness]
    //defining the stiffness matrix: d1xd2xd3 elements
    matrix_storage_info_t meta_(d1,d2,d3,fe::basis_cardinality(),fe::basis_cardinality());
    matrix_type stiffness_(meta_, 0.);
    //![instantiation_stiffness]

    /** defining the computation, i.e. for all elements:
        - computing the jacobian
        - computing its determinant
        - computing the jacobian inverse
        - integrate the stiffness matrix
        - adding the fluxes contribution
    */


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
    typedef gt::arg<8, typename discr_t::grad_storage_t> p_dphi;
    // defining the placeholder for the mass matrix values
    typedef gt::arg<9, matrix_type> p_stiffness;

    // appending the placeholders to the list of placeholders already in place
    typedef boost::mpl::vector<p_grid_points, p_jac, p_weights
                               , p_jac_det, p_jac_inv, p_phi_geo
                               , p_dphi_geo, p_phi, p_dphi, p_stiffness
                               > arg_list;
    //![placeholders]

    gt::aggregator_type<arg_list> domain(
        (p_grid_points() = assembler_base.grid())
        , (p_jac() = assembler.jac())
        , (p_weights() = assembler.fe_backend().cub_weights())
        , (p_jac_det() = assembler.jac_det())
        , (p_jac_inv() = assembler.jac_inv())
        , (p_phi_geo() = assembler.fe_backend().val())
        , (p_dphi_geo() = assembler.fe_backend().grad())
        , (p_phi() = fe_.val())
        , (p_dphi() = fe_.grad())
        , (p_stiffness() = stiffness_)
        );

    // , m_domain(boost::fusion::make_vector(&m_grid, &m_jac, &m_fe_backend.cub_weights(), &m_jac_det, &m_jac_inv, &m_fe_backend.local_gradient(), &m_fe_bac
                                                                                                   // , &m_stiffness, &m_assembled_stiffness
    auto coords=gt::grid<axis>({0, 0, 0, d1-1, d1},
        {0, 0, 0, d2-1, d2});
    coords.value_list[0] = 0;
    coords.value_list[1] = d3-1;

    //![computation]
    auto computation=gt::make_computation<BACKEND>(
        domain, coords,
        make_multistage
        (
            execute<forward>(),
            gt::make_stage<functors::update_jac<geo_t> >( p_grid_points(), p_dphi(), p_jac())
            ,
            gt::make_stage<functors::det<geo_t> >(p_jac(), p_jac_det())
            , gt::make_stage<functors::inv<geo_t> >(p_jac(), p_jac_det(), p_jac_inv())
            , gt::make_stage<functors::stiffness<fe, cub> >(p_jac_det(), p_jac_inv(), p_weights(), p_dphi(), p_dphi(), p_stiffness())//stiffness
            ));

    computation->ready();
    computation->steady();
    computation->run();
    computation->finalize();
    //![computation]

    return test(assembler_base, assembler, fe_, stiffness_)==true;
}
