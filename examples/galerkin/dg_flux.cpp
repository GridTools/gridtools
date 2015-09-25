/**
\file
*/
#pragma once
#define PEDANTIC_DISABLED
//! [assembly]
#include "bd_assembly.hpp"
//! [assembly]
#include "test_dg_flux.hpp"


// [boundary integration]
/**
   This functor computes an integran over a boundary face
*/

template <typename FE, typename BoundaryCubature>
struct boundary_integral {
    using fe=FE;
    using bd_cub=BoundaryCubature;

    using jac=accessor< 0, range<0,0,0,0>, 6 >;
    using jac_det=accessor< 1, range<0,0,0,0>, 4 >;
    using weights=accessor< 2, range<0,0,0,0>, 3 >;
    using phi_trace=accessor< 3, range<0,0,0,0>, 3 >;
    using normals=accessor< 4, range<0,0,0,0>, 5 >;
    using out=accessor< 5, range<0,0,0,0>, 5 >;
    using arg_list=boost::mpl::vector<jac, jac_det, weights, phi_trace, normals, out> ;

    /** @brief compute the integral on the boundary of a field times the normals

        note that we use here the traces of the basis functions, i.e. the basis functions
        evaluated on the quadrature points of the boundary faces.
    */
    template <typename Evaluation>
    GT_FUNCTION
    static void Do(Evaluation const & eval, x_interval) {
        x::Index i;
        y::Index j;
        z::Index k;
        Dimension<4>::Index quad;
        Dimension<5>::Index dimI;
        Dimension<6>::Index dimJ;
        Dimension<4>::Index dofI;
        Dimension<5>::Index dofJ;

        for(short_t P_i=0; P_i<fe::basisCardinality; ++P_i) // current dof
        {
            for(short_t P_j=0; P_j<fe::basisCardinality; ++P_j) // current dof
            {
                float_type partial_sum=0.;
                for(ushort_t q_=0; q_<bd_cub::numCubPoints; ++q_){
                    //loop on the basis functions (interpolation in the quadrature point)
                    float_type inner_product1=0.;
                    for(ushort_t j_=0; j_<3; ++j_){
                        inner_product1 += eval(normals(quad+q_, dimJ+j_)*jac(quad+q_)*!phi_trace(dofJ+P_i, quad+q_));
                    }
                    partial_sum += (inner_product1) * eval(!weights(quad+q_))*eval(jac_det(quad+q_));
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
    using matrix_storage_info_t=storage_info< gridtools::layout_map<0,1,2,3,4> >;
    using matrix_type=storage_t< matrix_storage_info_t >;
    using fe=reference_element<1, Lagrange, Hexa>;
    using geo_map=reference_element<1, Lagrange, Hexa>;
    using cub=cubature<fe::order, fe::shape>;
    using geo_t = intrepid::geometry<geo_map, cub>;
    // using discr_t = intrepid::discretization<fe, cub>;

    //boundary

    using bd_cub_t = intrepid::boundary_cub<geo_map, cub::cubDegree>;
    using bd_discr_t = intrepid::boundary_discr<bd_cub_t>;
    bd_cub_t bd_cub_;
    bd_discr_t bd_discr_(bd_cub_, 0);

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
                    // std::cout<<"grid point("<<m_grid( i,  j,  k,  point,  0) << ", "<< m_grid( i,  j,  k,  point,  1)<<", "<<m_grid( i,  j,  k,  point,  2)<<")"<<std::endl;
                }
    //![grid]

    //![instantiation_stiffness]
    //defining the stiffness matrix: d1xd2xd3 elements
    matrix_type flux_(d1,d2,d3,fe::basisCardinality,fe::basisCardinality);

    //![placeholders]
    // defining the placeholder for the flux
    typedef arg<as::size, matrix_type> p_flux;
    // defining the placeholder for the local gradient of the element boundary face
    typedef arg<as::size+1, bd_discr_t::grad_storage_t> p_bd_dphi;

    typedef arg<as::size+2, bd_discr_t::phi_storage_t> p_bd_phi;

    // appending the placeholders to the list of placeholders already in place
    auto domain=assembler.template domain<p_flux, p_bd_dphi, p_bd_phi>(flux_, bd_discr_.local_gradient(), bd_discr_.phi());
    //![placeholders]


    // , m_domain(boost::fusion::make_vector(&m_grid, &m_jac, &m_fe_backend.cub_weights(), &m_jac_det, &m_jac_inv, &m_fe_backend.local_gradient(), &m_fe_bac
    // , &m_stiffness, &m_assembled_stiffness
    auto coords=coordinates<axis>({1, 0, 1, d1-1, d1},
        {1, 0, 1, d2-1, d2});
    coords.value_list[0] = 0;
    coords.value_list[1] = d3-1;

    //![computation]
    auto computation=make_computation<gridtools::BACKEND, gridtools::layout_map<0,1,2,3> >(
        make_mss
        (
            execute<forward>()
            // evaluate the cell Jacobian matrix on the boundary (given the basis functions derivatives in those points)
            , make_esf<functors::update_jac<bd_discr_t , enumtype::Quad> >(as::p_grid_points(), as::p_bd_jac(), p_bd_dphi())
            // compute the normals on the quad points from the jacobian above (first 2 columns)
            , make_esf<functors::compute_face_normals<bd_discr_t> >(as::p_bd_jac(), as::p_normals())
            // surface integral:
            // compute the measure for the surface integral
            //            |  / d(phi_x)/du   d(phi_x)/dv  1 \  |
            //   det(J) = | |  d(phi_y)/du   d(phi_y)/dv  1  | |
            //            |  \ d(phi_z)/du   d(phi_z)/dv  1 /  |
            , make_esf<functors::measure<bd_discr_t, 2> >(as::p_bd_jac(), as::p_bd_measure())
            // evaluate the integral
            , make_esf<boundary_integral<fe, bd_cub_t::bd_cub> >(as::p_bd_jac(), as::p_bd_measure(), as::p_bd_weights(), p_bd_phi(), as::p_normals(), p_flux()) //flux
            ), domain, coords);

    computation->ready();
    computation->steady();
    computation->run();
    computation->finalize();
    //![computation]
    intrepid::test(assembler, bd_discr_, flux_);
}
