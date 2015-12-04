
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
// #include "test_dg_flux.hpp"
#include "../functors/dg_fluxes.hpp"
#include "../functors/matvec.hpp"

/**
   @brief flux F(u)

   in the equation \f$ \frac{\partial u}{\partial t}=F(u) \f$
*/
struct flux {
    template<typename Arg>
    GT_FUNCTION
    constexpr auto operator()(Arg const& arg_) -> decltype((Arg()+Arg())/2.){
        return (arg_+arg_)/2.;
    }
};

/**
@brief advection vector v

*/
struct advection_vector {
    static constexpr array<double, 3> value={1.,1.,1.};
};

constexpr array<double, 3> advection_vector::value;

int main(){
    //![definitions]
    using namespace enumtype;
    using namespace gridtools;
    //defining the assembler, based on the Intrepid definitions for the numerics
    using matrix_storage_info_t=storage_info< layout_tt<3,4>,  __COUNTER__ >;
    using matrix_type=storage_t< matrix_storage_info_t >;

    using geo_map=reference_element<1, Lagrange, Hexa>;
    using cub=cubature<geo_map::order, geo_map::shape>;
    using geo_t = intrepid::geometry<geo_map, cub>;

    geo_t fe_;
    fe_.compute(Intrepid::OPERATOR_GRAD);
    fe_.compute(Intrepid::OPERATOR_VALUE);

    //boundary
    using bd_cub_t = intrepid::boundary_cub<geo_map, cub::cubDegree>;
    using bd_discr_t = intrepid::boundary_discr<bd_cub_t>;
    bd_cub_t bd_cub_;

    bd_discr_t bd_discr_(bd_cub_, 0, 1, 2, 3, 4, 5);//face ordinals

    bd_discr_.compute(Intrepid::OPERATOR_GRAD);

    //![boundary]

    using as_base=assembly_base<geo_t>;
    using as=assembly<geo_t>;
    using as_bd=bd_assembly<bd_discr_t>;

    //![definitions]

    //dimensions of the problem (in number of elements per dimension)
    auto d1=8;
    auto d2=8;
    auto d3=1;

    geo_t geo_;
    geo_.compute(Intrepid::OPERATOR_GRAD);
    geo_.compute(Intrepid::OPERATOR_VALUE);
    //![as_instantiation]
    //constructing the integration tools on the boundary


    as_base assembler_base(d1,d2,d3);
    as assembler(geo_,d1,d2,d3);
    as_bd bd_assembler(bd_discr_,d1,d2,d3);

    using domain_tuple_t = domain_type_tuple<as_bd, as, as_base>;
    domain_tuple_t domain_tuple_ (bd_assembler, assembler, assembler_base);
    //![as_instantiation]

    //![grid]
    //constructing a structured cartesian grid
    for (uint_t i=0; i<d1; i++)
        for (uint_t j=0; j<d2; j++)
            for (uint_t k=0; k<d3; k++)
                for (uint_t point=0; point<geo_map::basisCardinality; point++)
                {
                    assembler_base.grid()( i,  j,  k,  point,  0)= (i + geo_.grid()(point, 0));
                    assembler_base.grid()( i,  j,  k,  point,  1)= (j + geo_.grid()(point, 1));
                    assembler_base.grid()( i,  j,  k,  point,  2)= (k + geo_.grid()(point, 2));
                }
    //![grid]

    //![instantiation_stiffness]
    //defining the advection matrix: d1xd2xd3 elements
    matrix_storage_info_t meta_(d1,d2,d3,geo_map::basisCardinality,geo_map::basisCardinality);
    matrix_type advection_(meta_, 0., "advection");
    matrix_type mass_(meta_, 0., "mass");

    using vector_storage_info_t=storage_info< layout_tt<3>,  __COUNTER__ >;//TODO change: iterate on faces
    using vector_type=storage_t< vector_storage_info_t >;

    vector_storage_info_t vec_meta_(d1,d2,d3,geo_map::basisCardinality);
    vector_type u_(vec_meta_, 2., "u");//initial solution
    vector_type result_(vec_meta_, 0., "result");//new solution

    //![placeholders]
    // defining the placeholder for the mass
    // typedef arg<domain_tuple_t::size, bd_matrix_type> p_bd_mass;
    // defining the placeholder for the local gradient of the element boundary face
    // typedef arg<domain_tuple_t::size+1, bd_discr_t::grad_storage_t> p_bd_dphi;

    // typedef arg<domain_tuple_t::size+2, bd_discr_t::basis_function_storage_t> p_bd_phi;
    typedef arg<domain_tuple_t::size, vector_type> p_u;
    typedef arg<domain_tuple_t::size+1, vector_type> p_result;
    typedef arg<domain_tuple_t::size+2, matrix_type> p_mass;
    typedef arg<domain_tuple_t::size+3, matrix_type> p_advection;
    typedef arg<domain_tuple_t::size+4, typename geo_t::basis_function_storage_t> p_phi;
    typedef arg<domain_tuple_t::size+5,  typename geo_t::grad_storage_t> p_dphi;

    // appending the placeholders to the list of placeholders already in place
    auto domain=domain_tuple_.template domain
        <p_u, p_result , p_mass, p_advection, p_phi, p_dphi>
        ( u_, result_, mass_, advection_, geo_.val(), geo_.grad());
    //![placeholders]


    auto coords=coordinates<axis>({1, 0, 1, d1-1, d1},
        {1, 0, 1, d2-1, d2});
    coords.value_list[0] = 0;
    coords.value_list[1] = d3-1;

    //short notation
    using dt=domain_tuple_t;

    //![computation]
    auto computation=make_computation< gridtools::BACKEND >(
        make_mss
        (
            execute<forward>()

            // boundary fluxes

            //computes the jacobian in the boundary points of each element
            , dt::update_bd_jac<enumtype::Hexa>::esf()
            //computes the measure of the boundaries with codimension 1 (ok, faces)
            , dt::measure<1>::esf()
            //computes the mass on the element boundaries
            , dt::bd_mass::esf()

            // Internal element

            //compute the Jacobian matrix
            , dt::update_jac<enumtype::Hexa>::esf()
            // compute the measure (det(J))
            , make_esf<functors::det<geo_t> >(dt::p_jac(), dt::p_jac_det())
            // compute the mass matrix
            , dt::mass< geo_t, cub >::esf(p_phi(), p_mass()) //mass
            // compute the advection matrix
            , dt::advection< geo_t, cub, advection_vector >::esf(p_phi(), p_dphi(), p_advection()) //advection

            // computing flux/discretize

            // initialize result=0
            , make_esf< functors::assign<4,int,0> >( p_result() )
            // compute Lax-Friedrich flux (communication-gather) result=flux;
            , dt::lax_friedrich<flux>::esf(p_u(), p_result())
            // integrate the flux: result=M_bd*flux
            , make_esf< functors::matvec_bd >( p_result(), dt::p_bd_mass(), p_result() )
            // result+=M*u
            , make_esf< functors::matvec >( p_u(), p_mass(), p_result() )
            // result+=A*u
            , make_esf< functors::matvec >( p_u(), p_advection(), p_result() )
            // Optional: assemble the result vector by summing the values on the element boundaries
            , make_esf< functors::assemble<geo_t, add_functor> >( p_result(), p_result(), p_result() )
            // , make_esf< time_advance >(p_u(), p_result())
            ), domain, coords);

    computation->ready();
    computation->steady();
    int T = 4;

    for(int i=0; i<T; ++i){
        computation->run();
        //simple first order time discretization
        u_.swap_pointers(result_);
    }
    computation->finalize();
    //![computation]
    // intrepid::test(assembler, bd_discr_, bd_mass_);
}
