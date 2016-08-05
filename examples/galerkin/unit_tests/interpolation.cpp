/**
\file
*/
#define PEDANTIC_DISABLED
#define HAVE_INTREPID_DEBUG

#include "../numerics/assembly.hpp"
#include "../functors/interpolate.hpp"

int main( int argc, char ** argv){

    //![definitions]
    using namespace gdl;
    using namespace gdl::enumtype;
    //defining the assembler, based on the Intrepid definitions for the numerics

    static const ushort_t order_geom=1;
    static const ushort_t order_discr=2;
    using geo_map=reference_element<order_geom, Lagrange, Hexa>;
    using discr_map=reference_element<order_discr, Legendre, Hexa>;
    // integrates exactly the composition of geo_map and basis functions polynomials
    using cub=cubature<4// (discr_map::order()*geo_map::order())+1
                       , geo_map::shape()>;
    using discr_t = intrepid::discretization<discr_map, cub>;
    using geo_t = intrepid::geometry<geo_map, cub>;

    discr_t fe_;
    geo_t geo_;
    fe_.compute(Intrepid::OPERATOR_VALUE);
    geo_.compute(Intrepid::OPERATOR_GRAD); // used to compute the Jacobian
    geo_.compute(Intrepid::OPERATOR_VALUE);

    using as_base=assembly_base<geo_t>;
    using as=assembly<geo_t>;

    //![definitions]

    //dimensions of the problem (in number of elements per dimension)
    auto d1=3;
    auto d2=3;
    auto d3=2;

    as_base assembler_base_(d1,d2,d3);
    as assembler_(geo_,d1,d2,d3);

    //![grid]
    //constructing a structured cartesian grid

    for (uint_t i=0; i<d1; i++)
        for (uint_t j=0; j<d2; j++)
            for (uint_t k=0; k<d3; k++)
                for (uint_t point=0; point<geo_map::basis_cardinality(); point++)
                {
                    assembler_base_.grid()( i,  j,  k,  point,  0)= (i + (1+geo_.grid()(point, 0, 0))/2.)/d1;
                    assembler_base_.grid()( i,  j,  k,  point,  1)= (j + (1+geo_.grid()(point, 1, 0))/2.)/d2;
                    assembler_base_.grid()( i,  j,  k,  point,  2)= (k + (1+geo_.grid()(point, 2, 0))/2.)/d3;
                }
    // ![grid]

    //![placeholders]
    // defining the placeholder for the mass

    using scalar_storage_info_t=storage_info< __COUNTER__, layout_tt<4> >;
    using scalar_type=storage_t< scalar_storage_info_t >;
    using interp_scalar_storage_info_t=storage_info< __COUNTER__, layout_tt<4> >;
    using interp_scalar_type = storage_t<interp_scalar_storage_info_t>;
    scalar_storage_info_t scalar_meta_(d1,d2,d3,cub::numCubPoints());
    interp_scalar_storage_info_t interp_scalar_meta_(d1,d2,d3,discr_map::basis_cardinality());
    scalar_type u_(scalar_meta_, 0., "u");//initial solution
    interp_scalar_type interp_u_(interp_scalar_meta_, 0., "interpolated u");
    scalar_type out_(scalar_meta_, 0., "final u");//initial solution

    auto grid_=gt::grid<axis>({1u, 0u, 1u, (uint_t)d1-1, (uint_t)d1},
        {1u, 0u, 1u, (uint_t)d2-1u, (uint_t)d2});
    grid_.value_list[0] = 1;
    grid_.value_list[1] = d3-1;

    //initialize with an affine function
    for (uint_t i=0; i<d1; i++)
        for (uint_t j=0; j<d2; j++)
            for (uint_t k=0; k<d3; k++)
                for (uint_t point=0; point<cub::numCubPoints(); point++)
                    u_(i,j,k,point)= 1.2+assembler_.fe_backend().cub_points()(point, 0, 0);

    struct assemble{
        typedef gt::arg<0, typename as_base::grid_type >       p_grid_points;
        typedef gt::arg<1, typename as::jacobian_type >        p_jac;
        typedef gt::arg<2, typename as::weights_storage_t >    p_weights;
        typedef gt::arg<3, typename as::storage_type >         p_jac_det;
        typedef gt::arg<4, typename as::grad_storage_t>        p_dphi;
    };

    typedef typename boost::mpl::vector< typename assemble::p_grid_points, typename assemble::p_jac, typename assemble::p_weights, typename assemble::p_jac_det, typename assemble::p_dphi> mpl_list_assemble;

    gt::aggregator_type<mpl_list_assemble> domain_assemble_(boost::fusion::make_vector(  &assembler_base_.grid()
                                                                                     , &assembler_.jac()
                                                                                     , &assembler_.cub_weights()
                                                                                     , &assembler_.jac_det()
                                                                                     , &assembler_.fe_backend().grad()
                                                            )
        );

    auto compute_assembly=gt::make_computation< BACKEND >(
        domain_assemble_, grid_
        , gt::make_multistage(
            execute<forward>()
            //compute the Jacobian matrix
            , gt::make_stage<functors::update_jac<as::geometry_t, Hexa> >(assemble::p_grid_points(), assemble::p_dphi(), assemble::p_jac())
            // compute the measure (det(J))
            , gt::make_stage<functors::det<geo_t> >(assemble::p_jac(), assemble::p_jac_det())
            )
        );

    compute_assembly->ready();
    compute_assembly->steady();
    compute_assembly->run();
    compute_assembly->finalize();

    struct transform {
        typedef  gt::arg<0, typename as::storage_type >    p_jac_det;
        typedef  gt::arg<1, typename as::weights_storage_t >   p_weights;
        typedef  gt::arg<2, typename discr_t::basis_function_storage_t> p_phi;
        typedef  gt::arg<3, scalar_type > p_u;
        typedef  gt::arg<4, interp_scalar_type > p_u_interp;
    };

    typedef typename boost::mpl::vector< typename transform::p_jac_det, typename transform::p_weights, typename transform::p_phi, typename transform::p_u, typename transform::p_u_interp> mpl_list_transform;

    gt::aggregator_type<mpl_list_transform> domain_transform_(
        boost::fusion::make_vector(
            &assembler_.jac_det()
            ,&assembler_.fe_backend().cub_weights()
            ,&fe_.val()
            ,&u_
            ,&interp_u_
            ));

    auto transform_=gt::make_computation< BACKEND >(
        domain_transform_, grid_
        , gt::make_multistage(
            execute<enumtype::forward>()
            , gt::make_stage< functors::transform >( typename transform::p_jac_det(), typename transform::p_weights(), typename transform::p_phi(), typename transform::p_u(), typename transform::p_u_interp() )
            )
        );

    transform_->ready();
    transform_->steady();
    transform_->run();
    transform_->finalize();

    std::cout<<"INITIAL VALUES: \n\n\n";

    for (uint_t i=1; i<d1; i++)
        for (uint_t j=1; j<d2; j++)
            for (uint_t k=1; k<d3; k++){
                for (uint_t point=0; point<cub::numCubPoints(); point++)
                {
                    std::cout<<u_(i,j,k,point)<<" ";
                }
                std::cout<<"\n";
            }

    std::cout<<"INTERPOLATED VALUES: \n\n\n";

    for (uint_t i=1; i<d1; i++)
        for (uint_t j=1; j<d2; j++)
            for (uint_t k=1; k<d3; k++){
                for (uint_t point=0; point<discr_t::fe::basis_cardinality(); point++)
                {
                    std::cout<<interp_u_(i,j,k,point)<<" ";
                }
                std::cout<<"\n";
            }

    struct counter_transform {

        typedef  gt::arg<0, typename discr_t::basis_function_storage_t> p_phi;
        typedef  gt::arg<1, interp_scalar_type> p_u_interpolated;
        typedef  gt::arg<2, typename as::weights_storage_t >    p_weights;
        typedef  gt::arg<3, typename as::storage_type >    p_jac_det;
        typedef  gt::arg<4, scalar_type> p_u;
    };

    typedef typename boost::mpl::vector< counter_transform::p_phi, counter_transform::p_u_interpolated, counter_transform::p_weights, counter_transform::p_jac_det, counter_transform::p_u > mpl_list_interp;

    gt::aggregator_type<mpl_list_interp> domain_interp_(boost::fusion::make_vector(
                                                   &fe_.val()
                                                   ,&interp_u_
                                                   ,&assembler_.fe_backend().cub_weights()
                                                   ,&assembler_.jac_det()
                                                   ,&out_
                                                   ));

    auto interpolation_=gt::make_computation< BACKEND >(
        domain_interp_, grid_
        , gt::make_multistage(
            execute<forward>()
            , gt::make_stage< functors::evaluate >( counter_transform::p_phi(), counter_transform::p_u_interpolated(), counter_transform::p_weights(), counter_transform::p_jac_det(),  counter_transform::p_u() )
            )
        );

    interpolation_->ready();
    interpolation_->steady();
    interpolation_->run();
    interpolation_->finalize();

    std::cout<<"FINAL VALUES: \n\n\n";

    bool success = true;
    for (uint_t i=1; i<d1; i++)
        for (uint_t j=1; j<d2; j++)
            for (uint_t k=1; k<d3; k++){
                for (uint_t point=0; point<cub::numCubPoints(); point++)
                {
                    if(out_(i,j,k,point) != u_(i,j,k,point))
                    {
                        success=false;
                        std::cout<<"ERROR in ["<<point<<"]: "<< out_(i,j,k,point) <<" != "<< u_(i,j,k,point)<<"\n";
                    }
                    std::cout<<out_(i,j,k,point)<<" ";
                }
                std::cout<<"\n";
            }

    assert(success);
}
