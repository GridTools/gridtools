/**
\file
*/
#define PEDANTIC_DISABLED
#define HAVE_INTREPID_DEBUG
//! [assembly]
#include "../functors/bd_assembly.hpp"
//! [assembly]
// #include "test_dg_flux.hpp"
#include "../functors/dg_fluxes.hpp"
#include "../functors/matvec.hpp"


// [boundary integration]
/**
   This functor computes an integran over a boundary face
*/

using namespace expressions;
template <typename FE, typename BoundaryCubature>
struct integration {
    using fe=FE;
    using bd_cub=BoundaryCubature;

    using jac_det=accessor< 0, range<0,0,0,0>, 5 >;
    using weights=accessor< 1, range<0,0,0,0>, 3 >;
    using phi_trace=accessor< 2, range<0,0,0,0>, 3 >;
    using psi_trace=accessor< 3, range<0,0,0,0>, 3 >;
    using out=accessor< 4, range<0,0,0,0>, 6 >;

    using arg_list=boost::mpl::vector<jac_det, weights, phi_trace, psi_trace, out> ;

    /** @brief compute the integral on the boundary of a field times the normals

        note that we use here the traces of the basis functions, i.e. the basis functions
        evaluated on the quadrature points of the boundary faces.
    */
    template <typename Evaluation>
    GT_FUNCTION
    static void Do(Evaluation const & eval, x_interval) {
        dimension<4>::Index quad;
        dimension<4>::Index dofI;
        dimension<5>::Index dofJ;

        uint_t const num_cub_points=eval.get().get_storage_dims(jac_det())[3];
        uint_t const basis_cardinality = eval.get().get_storage_dims(phi_trace())[0];
        uint_t const n_faces = eval.get().get_storage_dims(jac_det())[4];


        for(short_t face_=0; face_<n_faces; ++face_) // current dof
        {
            //loop on the basis functions (interpolation in the quadrature point)
            //over the whole basis TODO: can be reduced
            for(short_t P_i=0; P_i<basis_cardinality; ++P_i) // current dof
            {
                for(short_t P_j=0; P_j<basis_cardinality; ++P_j) // current dof
                {
                    float_type partial_sum=0.;
                    for(ushort_t q_=0; q_<num_cub_points; ++q_){
                        partial_sum += eval(!phi_trace(P_i,q_,face_)*!psi_trace(P_j, q_, face_)*jac_det(quad+q_, dimension<5>(face_)) * !weights(q_));
                    }
                    eval(out(dofI+P_i, dofJ+P_j, dimension<6>(face_)))=partial_sum;
                }
            }
        }
    }
};
// [boundary integration]

struct flux{
    template<typename Arg>
    constexpr auto operator()(Arg const& arg_) -> decltype((Arg()+Arg())/2.){
        return (arg_+arg_)/2.;
    }
};

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
    using bd_matrix_storage_info_t=storage_info< layout_tt<3,4,5>,  __COUNTER__ >;
    using bd_matrix_type=storage_t< bd_matrix_storage_info_t >;
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

    /**overdimensioned. Reduce*/
    bd_matrix_storage_info_t bd_meta_(d1,d2,d3,geo_map::basisCardinality,geo_map::basisCardinality, 6/*faces*/);
    bd_matrix_type bd_mass_(bd_meta_, 0., "mass");

    using vector_storage_info_t=storage_info< layout_tt<3>,  __COUNTER__ >;
    using vector_type=storage_t< vector_storage_info_t >;
    vector_storage_info_t vec_meta_(d1,d2,d3,geo_map::basisCardinality);
    vector_type u_(vec_meta_, 2., "u");//initial solution
    vector_type flux_(vec_meta_, 0., "flux");
    vector_type result_(vec_meta_, 0., "result");

    //![placeholders]
    // defining the placeholder for the mass
    typedef arg<domain_tuple_t::size, bd_matrix_type> p_bd_mass;
    // defining the placeholder for the local gradient of the element boundary face
    typedef arg<domain_tuple_t::size+1, bd_discr_t::grad_storage_t> p_bd_dphi;

    typedef arg<domain_tuple_t::size+2, bd_discr_t::basis_function_storage_t> p_bd_phi;
    typedef arg<domain_tuple_t::size+3, vector_type> p_u;
    typedef arg<domain_tuple_t::size+4, vector_type> p_flux;
    typedef arg<domain_tuple_t::size+5, vector_type> p_result;

    // appending the placeholders to the list of placeholders already in place
    auto domain=domain_tuple_.template domain<p_bd_mass, p_bd_dphi, p_bd_phi, p_u, p_flux, p_result>(bd_mass_, bd_discr_.grad(), bd_discr_.val(), u_,  flux_, result_);
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
            , make_esf<functors::update_bd_jac<bd_discr_t , enumtype::Hexa> >(dt::p_grid_points(), p_bd_dphi(), dt::p_bd_jac())
            , make_esf<functors::measure<bd_discr_t, 2> >(dt::p_bd_jac(),
                                                          dt::p_bd_measure())
            , make_esf<integration<geo_map, bd_cub_t::bd_cub> >(dt::p_bd_measure(), dt::p_bd_weights(), p_bd_phi(), p_bd_phi(), p_bd_mass()) //mass
            //, make_esf<functors::bassi_rebay<bd_discr_t> >(p_u(), p_u(), p_flux())
            , make_esf<functors::lax_friedrich<bd_discr_t, flux > >(p_u(), p_u(), p_flux())

            // // Internal element
            // , make_esf<functors::update_jac<geo_t , enumtype::Hexa> >(as::p_grid_points(), p_dphi(), as::p_jac())
            // , make_esf<functors::det<geo_t> >(as::p_jac(), as::p_jac_det())
            // , make_esf<advection<fe> >(as::p_jac_det(), as::p_weight(), p_phi(), p_dphi(), p_advection()) //advection
            // // integrate the two contributions
            // , make_esf< integrate >( p_u(), p_flux(), p_bd_mass(), p_advection(), p_result() )
            // , make_esf< time_advance >(p_u(), p_result())
            ), domain, coords);

    computation->ready();
    computation->steady();
    int T = 4;

    for(int i=0; i<T; ++i){
        computation->run();
        // computation->cycle();
    }
    computation->finalize();
    //![computation]
    // intrepid::test(assembler, bd_discr_, bd_mass_);
}
