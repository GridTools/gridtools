//this MUST be included before any boost include
#define FUSION_MAX_VECTOR_SIZE 40
#define FUSION_MAX_MAP_SIZE FUSION_MAX_VECTOR_SIZE
#define BOOST_MPL_LIMIT_VECTOR_SIZE FUSION_MAX_VECTOR_SIZE
#define BOOST_MPL_CFG_NO_PREPROCESSED_HEADERS


/**
\file
*/
#define PEDANTIC_DISABLED
#define HAVE_INTREPID_DEBUG

//! [assembly]
#include "../numerics/bd_assembly.hpp"
//! [assembly]
#include "../numerics/tensor_product_element.hpp"

int main( int argc, char ** argv){

    if(argc!=2){
        printf("usage: \n >> legendre <N>\n");
        exit(-666);
    }
    int it_ = atoi(argv[1]);

    //![definitions]
    using namespace gridtools;
    using namespace gdl;
    using namespace gdl::enumtype;
    //defining the assembler, based on the Intrepid definitions for the numerics
    using matrix_storage_info_t=storage_info< __COUNTER__, layout_tt<3,4> >;
    using matrix_type=storage_t< matrix_storage_info_t >;

    static const ushort_t order_geom=1; //affine geometric maps
    static const ushort_t order_discr=2; //order 2 polynomials for the FE space
    using geo_map=reference_element<order_geom, Lagrange, Hexa>;
    using discr_map=reference_element<order_discr, Legendre, Hexa>;
    //integrate exactly polyunomials of degree (discr_map::order*geo_map::order)
    using cub=cubature<(discr_map::order() * geo_map::order())+2, geo_map::shape()>;//overintegrating: few basis func are 0 on all quad points otherwise...
    using discr_t = intrepid::discretization<discr_map, cub>;
    using geo_t = intrepid::geometry<geo_map, cub>;

    discr_t fe_;
    geo_t geo_;
    fe_.compute(Intrepid::OPERATOR_VALUE);
    geo_.compute(Intrepid::OPERATOR_GRAD); // used to compute the Jacobian

    using as_base=assembly_base<geo_t>;
    using as=assembly<geo_t>;

    //![definitions]

    //dimensions of the problem (in number of elements per dimension)
    auto d1=4;
    auto d2=4;
    auto d3=4;

    as_base assembler_base(d1,d2,d3);
    as assembler(geo_,d1,d2,d3);

    //![grid]
    //constructing a structured cartesian grid

    for (uint_t i=0; i<d1; i++)
        for (uint_t j=0; j<d2; j++)
            for (uint_t k=0; k<d3; k++)
                for (uint_t point=0; point<geo_map::basis_cardinality(); point++)
                {
                    assembler_base.grid()( i,  j,  k,  point,  0)= (i + (1+geo_.grid()(point, 0, 0))/2.)/d1;
                    assembler_base.grid()( i,  j,  k,  point,  1)= (j + (1+geo_.grid()(point, 1, 0))/2.)/d2;
                    assembler_base.grid()( i,  j,  k,  point,  2)= (k + (1+geo_.grid()(point, 2, 0))/2.)/d3;
                }
    // ![grid]

    //![instantiation_mass]
    //defining the mass matrix: d1xd2xd3 elements
    matrix_storage_info_t meta_(d1,d2,d3,discr_map::basis_cardinality(),discr_map::basis_cardinality());
    matrix_type mass_(meta_, 0., "mass");

    //![placeholders]
    // defining the placeholder for the mass


    typedef arg<0, typename as_base::grid_type >       p_grid_points;
    typedef arg<1, typename as::jacobian_type >   p_jac;
    typedef arg<2, typename as::geometry_t::weights_storage_t >   p_weights;
    typedef arg<3, typename as::storage_type >    p_jac_det;
    typedef arg<4, typename as::jacobian_type >   p_jac_inv;
    // typedef arg<5, typename as::geometry_t::basis_function_storage_t> p_phi;
    typedef arg<5, typename as::geometry_t::grad_storage_t> p_dphi;

    typedef arg<6, matrix_type> p_mass;
    typedef arg<7, typename discr_t::basis_function_storage_t> p_phi_discr;

    typedef typename boost::mpl::vector<p_grid_points, p_jac, p_weights, p_jac_det, p_jac_inv,  p_dphi, p_mass, p_phi_discr > mpl_list;

    aggregator_type<mpl_list> domain(boost::fusion::make_vector(  &assembler_base.grid()
                                                              , &assembler.jac()
                                                              , &assembler.fe_backend().cub_weights()
                                                              , &assembler.jac_det()
                                                              , &assembler.jac_inv()
                                                              , &assembler.fe_backend().grad()
                                                              , &mass_
                                                              , &fe_.val()
                                     ));

    //![placeholders]

    auto coords=grid<axis>({1u, 0u, 1u, (uint_t)d1-1, (uint_t)d1},
        {1u, 0u, 1u, (uint_t)d2-1u, (uint_t)d2});
    coords.value_list[0] = 1;
    coords.value_list[1] = d3-1;

    //![computation]
    auto compute_assembly=make_computation< BACKEND >(
        domain, coords,
        make_multistage
        (
            execute<forward>()

            //compute the Jacobian matrix
            , make_stage<functors::update_jac<as::geometry_t, Hexa> >(p_grid_points(), p_dphi(), p_jac())
            // compute the measure (det(J))
            , make_stage<functors::det<geo_t> >(p_jac(), p_jac_det())
            // compute the mass matrix
            , make_stage< functors::mass >(p_jac_det(), p_weights(), p_phi_discr(), p_phi_discr(), p_mass()) //mass
            ));

    compute_assembly->ready();
    compute_assembly->steady();
    compute_assembly->run();

    bool success=true;
    for (uint_t i=1; i<d1; i++)
        for (uint_t j=1; j<d2; j++)
            for (uint_t k=1; k<d3; k++){
                for (uint_t dof1=0; dof1<discr_map::basis_cardinality(); dof1++)
                    for (uint_t dof2=0; dof2<discr_map::basis_cardinality(); dof2++)
                    {
                        if(dof1 != dof2 && mass_(i,j,k,dof1,dof2)*mass_(i,j,k,dof1,dof2) > 1.e-10)
                        {
                            success=false;
                            std::cout<<"ERROR in ["<<dof1<<", "<<dof2<<"]: "<< mass_(i,j,k,dof1,dof2) <<" != 0\n";
                        }
                    }
            }

    assert(success);
}
