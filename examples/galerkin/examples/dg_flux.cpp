/**
\file
*/

//this MUST be included before any boost include
#define FUSION_MAX_VECTOR_SIZE 40
#define FUSION_MAX_MAP_SIZE FUSION_MAX_VECTOR_SIZE
#define BOOST_MPL_LIMIT_VECTOR_SIZE FUSION_MAX_VECTOR_SIZE
#define BOOST_MPL_CFG_NO_PREPROCESSED_HEADERS


#define PEDANTIC_DISABLED
#define HAVE_INTREPID_DEBUG
//! [assembly]
#include "../numerics/bd_assembly.hpp"
//! [assembly]
#include "test_dg_flux.hpp"
#include "../functors/dg_fluxes.hpp"
#include "../functors/matvec.hpp"


// [boundary integration]
/**
   This functor computes an integran over a boundary face
*/

namespace gdl{

    using namespace gt::expressions;
    template <typename FE, typename BoundaryCubature>
    struct integration {
        using fe=FE;
        using bd_cub=BoundaryCubature;

        using jac_det=gt::accessor< 0, enumtype::in, gt::extent<0,0,0,0>, 5 >;
        using weights=gt::accessor< 1, enumtype::in, gt::extent<0,0,0,0>, 4 >;
        using phi_trace=gt::accessor< 2, enumtype::in, gt::extent<0,0,0,0>, 3 >;
        using psi_trace=gt::accessor< 3, enumtype::in, gt::extent<0,0,0,0>, 3 >;
        using out=gt::accessor< 4, enumtype::inout, gt::extent<0,0,0,0>, 6 >;
        using arg_list=boost::mpl::vector<jac_det, weights, phi_trace, psi_trace, out> ;

        /** @brief compute the integral on the boundary of a field times the normals

            note that we use here the traces of the basis functions, i.e. the basis functions
            evaluated on the quadrature points of the boundary faces.
        */
        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_interval) {
            gt::dimension<4>::Index quad;
            gt::dimension<4>::Index dofI;
            gt::dimension<5>::Index dofJ;

            uint_t const num_cub_points=eval.template get_storage_dim<3>(jac_det());
            uint_t const basis_cardinality = eval.template get_storage_dim<3>(phi_trace());

            //NOTE: missing the loop on the faces for this example
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
}//namespace gdl

// [boundary integration]

int main(){
    //![definitions]
    using namespace gdl;
    using namespace enumtype;
    //defining the assembler, based on the Intrepid definitions for the numerics
    using matrix_storage_info_t=storage_info< __COUNTER__, layout_tt<3,4,5>>;
    using matrix_type=storage_t< matrix_storage_info_t >;

    using fe=reference_element<1, Lagrange, Hexa>;
    using geo_map=reference_element<1, Lagrange, Hexa>;
    using cub=cubature<fe::order(), fe::shape()>;
    using geo_t = intrepid::geometry<geo_map, cub>;

    //boundary
    using bd_cub_t = intrepid::boundary_cub<geo_map, cub::cubDegree>;
    using bd_discr_t = intrepid::boundary_discr<bd_cub_t, 1>;
    bd_cub_t bd_cub_;
    bd_discr_t bd_discr_(bd_cub_, 1);

    bd_discr_.compute(Intrepid::OPERATOR_GRAD);

    //![boundary]


    using as_base=assembly_base<geo_t>;
    using as=bd_assembly<bd_discr_t>;

    //![definitions]

    //dimensions of the problem (in number of elements per dimension)
    auto d1=8;
    auto d2=8;
    auto d3=1;

    geo_t geo;
    //![as_instantiation]
    //constructing the integration tools on the boundary
    as_base assembler_base(d1,d2,d3);
    as assembler(bd_discr_,d1,d2,d3);

    //![as_instantiation]

    //![grid]
    //constructing a structured cartesian grid
    for (uint_t i=0; i<d1; i++)
        for (uint_t j=0; j<d2; j++)
            for (uint_t k=0; k<d3; k++)
                for (uint_t point=0; point<fe::basis_cardinality(); point++)
                {
                    assembler_base.grid()( i,  j,  k,  point,  0)= (i + (1+geo.grid()(point, 0, 0))/2.)/d1;
                    assembler_base.grid()( i,  j,  k,  point,  1)= (j + (1+geo.grid()(point, 1, 0))/2.)/d2;
                    assembler_base.grid()( i,  j,  k,  point,  2)= (k + (1+geo.grid()(point, 2, 0))/2.)/d3;
                }
    //![grid]

    //![instantiation_stiffness]
    //defining the stiffness matrix: d1xd2xd3 elements
    matrix_storage_info_t meta_(d1,d2,d3,fe::basis_cardinality(),fe::basis_cardinality(), 1/*faces*/);
    matrix_type mass_(meta_, 0., "mass");

    using vector_storage_info_t=storage_info< __COUNTER__, layout_tt<3,4> >;
    using vector_type=storage_t< vector_storage_info_t >;
    vector_storage_info_t vec_meta_(d1,d2,d3,fe::basis_cardinality(), 1/*face*/);
    vector_type u_(vec_meta_, 2., "u");//initial solution
    vector_type jump_(vec_meta_, 0., "jump");
    vector_type flux_(vec_meta_, 0., "flux");
    vector_type integrated_flux_(vec_meta_, 0., "integrated flux");

    //![placeholders]
    typedef gt::arg<0, typename as::jacobian_type >       p_bd_jac;
    typedef gt::arg<1, typename as::face_normals_type >                   p_normals;
    typedef gt::arg<2, typename as::storage_type >        p_bd_measure;
    typedef gt::arg<3, typename as::boundary_t::weights_storage_t> p_bd_weights;
    typedef gt::arg<4, typename as::boundary_t::tangent_storage_t> p_ref_normals;
    typedef gt::arg<5, typename as::boundary_t::basis_function_storage_t> p_bd_phi;
    typedef gt::arg<6, typename as::boundary_t::grad_storage_t> p_bd_dphi;

    typedef gt::arg<7, typename as_base::grid_type >       p_grid_points;

    // defining the placeholder for the mass
    typedef gt::arg<8, matrix_type> p_mass;
    // defining the placeholder for the local gradient of the element boundary face
    typedef gt::arg<9, bd_discr_t::grad_storage_t> p_bd_discr_dphi;
    typedef gt::arg<10, bd_discr_t::basis_function_storage_t> p_bd_discr_phi;
    typedef gt::arg<11, vector_type> p_u;
    typedef gt::arg<12, vector_type> p_jump;
    typedef gt::arg<13, vector_type> p_flux;
    typedef gt::arg<14, vector_type> p_integrated_flux;

    typedef boost::mpl::vector<p_bd_jac, p_normals, p_bd_measure, p_bd_weights, p_ref_normals, p_bd_phi, p_bd_dphi, p_grid_points, p_mass, p_bd_discr_dphi, p_bd_discr_phi, p_u, p_jump, p_flux, p_integrated_flux> arg_list;

    // appending the placeholders to the list of placeholders already in place
    auto domain=gt::aggregator_type<arg_list>(
        (p_bd_jac()=assembler.bd_jac())
        , (p_normals()=assembler.normals())
        , (p_bd_measure()=assembler.bd_measure())
        , (p_bd_weights()=assembler.bd_backend().bd_cub_weights())
        , (p_ref_normals()=assembler.ref_normals())
        , (p_bd_phi()=assembler.bd_backend().val())
        , (p_bd_dphi()=assembler.bd_backend().grad())
        , (p_grid_points()=assembler_base.grid())
        , (p_mass()=mass_)
        , (p_bd_discr_dphi()=bd_discr_.grad())
        , (p_bd_discr_phi()=bd_discr_.val())
        , (p_u()=u_)
        , (p_jump()=jump_)
        , (p_flux()=flux_)
        , (p_integrated_flux()=integrated_flux_)
        );
    //![placeholders]

    auto coords=gt::grid<axis>({1, 0, 1, (uint_t)d1-1, (uint_t)d1},
        {1, 0, 1, (uint_t)d2-1, (uint_t)d2});
    coords.value_list[0] = 0;
    coords.value_list[1] = d3-1;

    //![computation]
    auto computation=gt::make_computation< BACKEND >(
        domain, coords,
        make_multistage
        (
            execute<forward>()
            // evaluate the cell Jacobian matrix on the boundary (given the basis functions derivatives in those points)
            , gt::make_stage<functors::update_bd_jac<bd_discr_t , enumtype::Hexa> >(p_grid_points(), p_bd_dphi(), p_bd_jac())
            // compute the normals on the quad points from the jacobian above, transporting the normals from the reference configuration, unnecessary. (can be computed also as the cross product of the first 2 columns of J)
            , gt::make_stage<functors::compute_face_normals<bd_discr_t> >(p_bd_jac(), p_ref_normals(), p_normals())
            // surface integral:
            // compute the measure for the surface integral
            //            |  / d(phi_x)/du   d(phi_x)/dv  1 \  |
            //   det(J) = | |  d(phi_y)/du   d(phi_y)/dv  1  | |
            //            |  \ d(phi_z)/du   d(phi_z)/dv  1 /  |
            , gt::make_stage<functors::measure<bd_discr_t, 1> >(p_bd_jac(), p_bd_measure())
            // evaluate the mass matrix
            , gt::make_stage<integration<fe, bd_cub_t::bd_cub> >(p_bd_measure(), p_bd_weights(), p_bd_discr_phi(), p_bd_discr_phi(), p_mass()) //mass
            // compute the flux, this line defines the type of approximation we choose for DG
            // (e.g. upwind, centered, Lax-Wendroff and so on)
            // NOTE: if not linear we cannot implement it as a matrix
            , gt::make_stage<functors::bassi_rebay<bd_discr_t> >(p_u(), p_u(), p_flux())
            // integrate the flux: mass*flux
            , gt::make_stage< functors::matvec_BdxBdxBd >( p_flux(), p_mass(), p_integrated_flux() )
            ));

    computation->ready();
    computation->steady();
    computation->run();
    computation->finalize();
    //![computation]
    gt::intrepid::template test<geo_t>(assembler_base, assembler, bd_discr_, mass_);
}
