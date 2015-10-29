/**
\file
*/
#define PEDANTIC_DISABLED
#define HAVE_INTREPID_DEBUG
//! [assembly]
#include "../functors/bd_assembly.hpp"
//! [assembly]
#include "test_dg_flux.hpp"
#include "../functors/dg_fluxes.hpp"
#include "../functors/matvec.hpp"


// [boundary integration]
/**
   This functor computes an integran over a boundary face
*/

template <typename FE, typename BoundaryCubature>
struct integration {
    using fe=FE;
    using bd_cub=BoundaryCubature;

    using jac_det=accessor< 0, range<0,0,0,0>, 4 >;
    using weights=accessor< 1, range<0,0,0,0>, 3 >;
    using out=accessor< 2, range<0,0,0,0>, 5 >;
    using phi_trace=accessor< 3, range<0,0,0,0>, 3 >;
    using psi_trace=accessor< 4, range<0,0,0,0>, 3 >;
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
        uint_t const basis_cardinality = eval.get().get_storage_dims(phi_trace())[3];

        //loop on the basis functions (interpolation in the quadrature point)
        for(short_t P_i=0; P_i<basis_cardinality; ++P_i) // current dof
        {
            for(short_t P_j=0; P_j<basis_cardinality; ++P_j) // current dof
            {
                float_type partial_sum=0.;
                for(ushort_t q_=0; q_<num_cub_points; ++q_){
                    partial_sum += eval(!phi_trace(P_i,q_)*!psi_trace(P_j, q_)*jac_det(quad+q_));
                }
                eval(out(dofI+P_i, dofJ+P_j))=partial_sum;
            }
        }
    }
};
// [boundary integration]

int main(){
    //![definitions]
    using namespace enumtype;
    using namespace gridtools;
    //defining the assembler, based on the Intrepid definitions for the numerics
    using matrix_storage_info_t=storage_info< layout_tt<3,4>,  __COUNTER__ >;
    using matrix_type=storage_t< matrix_storage_info_t >;

    using fe=reference_element<1, Lagrange, Hexa>;
    using geo_map=reference_element<1, Lagrange, Hexa>;
    using cub=cubature<fe::order, fe::shape>;
    using geo_t = intrepid::geometry<geo_map, cub>;

    //boundary
    using bd_cub_t = intrepid::boundary_cub<geo_map, cub::cubDegree>;
    using bd_discr_t = intrepid::boundary_discr<bd_cub_t>;
    bd_cub_t bd_cub_;
    bd_discr_t bd_discr_(bd_cub_, 1);

    bd_discr_.compute(Intrepid::OPERATOR_GRAD);

    //![boundary]


    using as=assembly<bd_discr_t, geo_t>;


    //![definitions]

    //dimensions of the problem (in number of elements per dimension)
    auto d1=8;
    auto d2=8;
    auto d3=1;

    geo_t geo;
    //![as_instantiation]
    //constructing the integration tools on the boundary
    as assembler(bd_discr_,d1,d2,d3);
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
    matrix_type mass_(meta_, 0., "mass");

    using vector_storage_info_t=storage_info< layout_tt<3>,  __COUNTER__ >;
    using vector_type=storage_t< vector_storage_info_t >;
    vector_storage_info_t vec_meta_(d1,d2,d3,fe::basisCardinality);
    vector_type u_(vec_meta_, 2., "u");//initial solution
    vector_type jump_(vec_meta_, 0., "jump");
    vector_type flux_(vec_meta_, 0., "flux");
    vector_type integrated_flux_(vec_meta_, 0., "integrated flux");

    //![placeholders]
    // defining the placeholder for the mass
    typedef arg<as::size, matrix_type> p_mass;
    // defining the placeholder for the local gradient of the element boundary face
    typedef arg<as::size+1, bd_discr_t::grad_storage_t> p_bd_dphi;

    typedef arg<as::size+2, bd_discr_t::basis_function_storage_t> p_bd_phi;
    typedef arg<as::size+3, vector_type> p_u;
    typedef arg<as::size+4, vector_type> p_jump;
    typedef arg<as::size+5, vector_type> p_flux;
    typedef arg<as::size+6, vector_type> p_integrated_flux;

    // appending the placeholders to the list of placeholders already in place
    auto domain=assembler.template domain<p_mass, p_bd_dphi, p_bd_phi, p_u, p_jump, p_flux, p_integrated_flux>(mass_, bd_discr_.grad(), bd_discr_.val(), u_, jump_, flux_, integrated_flux_);
    //![placeholders]


    auto coords=coordinates<axis>({1, 0, 1, d1-1, d1},
        {1, 0, 1, d2-1, d2});
    coords.value_list[0] = 0;
    coords.value_list[1] = d3-1;

    //![computation]
    auto computation=make_computation< gridtools::BACKEND >(
        make_mss
        (
            execute<forward>()
            // evaluate the cell Jacobian matrix on the boundary (given the basis functions derivatives in those points)
            , make_esf<functors::update_jac<bd_discr_t , enumtype::Hexa> >(as::p_grid_points(), p_bd_dphi(), as::p_bd_jac())
            // compute the normals on the quad points from the jacobian above (first 2 columns), unnecessary
            , make_esf<functors::compute_face_normals<bd_discr_t> >(as::p_bd_jac(), as::p_ref_normals(), as::p_normals())
            // surface integral:
            // compute the measure for the surface integral
            //            |  / d(phi_x)/du   d(phi_x)/dv  1 \  |
            //   det(J) = | |  d(phi_y)/du   d(phi_y)/dv  1  | |
            //            |  \ d(phi_z)/du   d(phi_z)/dv  1 /  |
            , make_esf<functors::measure<bd_discr_t, 2> >(as::p_bd_jac(), as::p_bd_measure())
            // evaluate the mass matrix
            , make_esf<integration<fe, bd_cub_t::bd_cub> >(as::p_bd_measure(), as::p_bd_weights(), p_mass(), p_bd_phi(), p_bd_phi()) //mass
            // compute the flux, this line defines the type of approximation we choose for DG
            // (e.g. upwind, centered, Lax-Wendroff and so on)
            // NOTE: if not linear we cannot implement it as a matrix
            , make_esf<functors::bassi_rebay<bd_discr_t> >(p_u(), p_u(), p_flux())
            // integrate the flux: mass*flux
            , make_esf< functors::matvec >( p_flux(), p_mass(), p_integrated_flux() )
            ), domain, coords);

    computation->ready();
    computation->steady();
    computation->run();
    computation->finalize();
    //![computation]
    intrepid::test(assembler, bd_discr_, mass_);
}
