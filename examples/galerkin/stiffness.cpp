/**
\file
*/
#pragma once
#define PEDANTIC_DISABLED
//! [assembly]
#include "assembly.h"
//! [assembly]
#include "test_assembly.h"

// [integration]
/** The following functor performs the assembly of an elemental laplacian.
*/
template <typename FE, typename Cubature>
struct stiffness {
    using fe=FE;
    using cub=Cubature;

    //![accessors]
    using jac_det =accessor<0, range<0,0,0,0> , 4> const;
    using jac_inv =accessor<1, range<0,0,0,0> , 6> const;
    using weights =accessor<2, range<0,0,0,0> , 3> const;
    using stiff   =accessor<3, range<0,0,0,0> , 5> ;
    using dphi    =accessor<4, range<0,0,0,0> , 3> const;
    using dpsi    =accessor<5, range<0,0,0,0> , 3> const;
    using arg_list= boost::mpl::vector<jac_det, jac_inv, weights, stiff, dphi,dpsi> ;
    //![accessors]

    //![Do_stiffness]
    template <typename Evaluation>
    GT_FUNCTION
    static void Do(Evaluation const & eval, x_interval) {

        //quadrature points dimension
        dimension<4>::Index qp;
        //dimension 'i' in the stiffness matrix
        dimension<5>::Index dimx;
        //dimension 'j' in the stiffness matrix
        dimension<6>::Index dimy;

        //loop on the basis functions
        for(short_t P_i=0; P_i<fe::basisCardinality; ++P_i) // current dof
        {
            //loop on the test functions
            for(short_t Q_i=0; Q_i<fe::basisCardinality; ++Q_i)
            {
                //loop on the cub points
                for(short_t q=0; q<cub::numCubPoints(); ++q){
                    //inner product of the gradients
                    double gradients_inner_product=0.;
                    for(short_t icoor=0; icoor< fe::spaceDim; ++icoor)
                    {
                        gradients_inner_product +=
                            eval((jac_inv(qp+q, dimx+0, dimy+icoor)*!dphi(P_i,q,(uint_t)0)+
                                  jac_inv(qp+q, dimx+1, dimy+icoor)*!dphi(P_i,q,(uint_t)1)+
                                  jac_inv(qp+q, dimx+2, dimy+icoor)*!dphi(P_i,q,(uint_t)2))
                                 *
                                 (jac_inv(qp+q, dimx+0, dimy+icoor)*!dphi(Q_i,q,(uint_t)0)+
                                  jac_inv(qp+q, dimx+1, dimy+icoor)*!dphi(Q_i,q,(uint_t)1)+
                                  jac_inv(qp+q, dimx+2, dimy+icoor)*!dphi(Q_i,q,(uint_t)2)));
                    }
                    //summing up contributions (times the measure and quad weight)
                    eval(stiff(0,0,0,P_i,Q_i)) += gradients_inner_product * eval(jac_det(qp+q)*!weights(q,0,0));
                }
            }
        }
    }
    //![Do_stiffness]
};
//[integration]


// template <typename FE, typename BoundaryCubature>
// struct boundary_integral {
//     using fe=FE;
//     using bd_cub=BoundaryCubature;

//     using jac=accessor< 0, range<0,0,0,0>, 6 >;
//     using jac_det=accessor< 1, range<0,0,0,0>, 4 >;
//     using weights=accessor< 2, range<0,0,0,0>, 3 >;
//     using phi_trace=accessor< 3, range<0,0,0,0>, 3 >;
//     using normals=accessor< 4, range<0,0,0,0>, 5 >;
//     using out=accessor< 5, range<0,0,0,0>, 5 >;
//     using arg_list=boost::mpl::vector<jac, jac_det, weights, phi_trace, normals, out> ;

//     /** @brief compute the integral on the boundary of a field times the normals

//         note that we use here the traces of the basis functions, i.e. the basis functions
//         evaluated on the quadrature points of the boundary faces.
//     */
//     template <typename Evaluation>
//     GT_FUNCTION
//     static void Do(Evaluation const & eval, x_interval) {
//         x::Index i;
//         y::Index j;
//         z::Index k;
//         dimension<4>::Index quad;
//         dimension<5>::Index dimI;
//         dimension<6>::Index dimJ;
//         dimension<4>::Index dofI;
//         dimension<5>::Index dofJ;

//         for(short_t P_i=0; P_i<fe::basisCardinality; ++P_i) // current dof
//         {
//             for(short_t P_j=0; P_j<fe::basisCardinality; ++P_j) // current dof
//             {
//                 float_type partial_sum=0.;
//                 for(ushort_t q_=0; q_<bd_cub::numCubPoints; ++q_){
//                     //loop on the basis functions (interpolation in the quadrature point)
//                     float_type inner_product1=0.;
//                     for(ushort_t j_=0; j_<3; ++j_){
//                         inner_product1 += eval(normals(quad+q_, dimJ+j_)*jac(quad+q_)*!phi_trace(dofJ+P_i, quad+q_));
//                     }
//                     partial_sum += (inner_product1) * eval(weights(quad+q_) *jac_det(quad+q_));
//                 }
//                 eval(out(dofI+P_i, dofJ+P_j))=partial_sum;
//             }
//         }
//     }
// };

















//     using fe=FE;
//     using bd_cub=BdCubature;

//     //the accessors are linked to the fields
//     using in=accessor<0, range<0,0,0,0> , 5> const;
//     using out=accessor<1, range<0,0,0,0> , 5> const;
//     using normals=<2, range<0,0,0,0> , 5> const;
//     using arg_list=boost::mpl::vector<in,out> ;

//     static int_t n_faces=geo_map::hypercube_t::n_boundary_w_dim<2>::value;
//     // points on the edges
//     static const int_t bd_dim=fe::hypercube_t::template boundary_w_dim<1>::n_points::value;
//     static const int_t bd_dim_square=bd_dim*bd_dim;
//     static const int_t one=1;

//     template<ushort_t D>
//     struct stride{
//         // number of dof on an edge in x direction -1 (to compute the value on an opposite face)
//         static const uint_t value=fe::layout_t::template find<D>(bd_dim_square, bd_dim, one);
//     };

//     template<ushort_t D>
//     struct end{
//         // number of dof on an edge in x direction -1 (to compute the value on an opposite face)
//         static const uint_t value=(stride<D>::value*(bd_dim-1));
//     };

//     template <typename Evaluation>
//     GT_FUNCTION
//     static void Do(Evaluation const & eval, x_interval) {
//         x::Index i;
//         y::Index j;
//         z::Index k;
//         Dimension<4>::Index point;

//         //Hypothesis: the local dofs are ordered according to fe::layout
//         array<int, 3> strides={bd_dim*bd_dim, bd_dim, 1};
//         //for all dofs in a boundary face
//         for(short_t I=0; I<bd_dim; I++)
//             for(short_t J=0; J<bd_dim; J++)
//             {
//                 //dof of the basis function
//                 auto dof_x=(stride<1>::value*I+stride<2>::value*J);
//                 //dof of the basis function
//                 auto dof_y=(stride<0>::value*I+stride<2>::value*J);

//                 //normal vector

//                 //jump on the face x=0
//                 auto jump_xminus=eval(in(point+dof_x)-in(i-1, point+(dof_x+end<0>::value)));
//                 //jump on the face x=bd_dim
//                 auto jump_xplus=eval(in(point+(dof_x+end<0>::value))-in(i+1, point+dof_x));
//                 //jump on the face y=0
//                 auto jump_yminus=eval(in(point+dof_y)-in(i-1, point+(dof_y+end<1>::value)));
//                 //jump on the face y=bd_dim
//                 auto jump_yplus=eval(in(point+(dof_y+end<1>::value))-in(i+1, point+dof_y));

//                 //jump on the face x=0
//                 auto mean_xminus=eval(in(point+dof_x)+in(i-1, point+(dof_x+end<0>::value)))/2.;
//                 //jump on the face x=bd_dim
//                 auto mean_xplus=eval(in(point+(dof_x+end<0>::value))+in(i+1, point+dof_x))/2.;
//                 //jump on the face y=0
//                 auto mean_yminus=eval(in(point+dof_y)+in(i-1, point+(dof_y+end<1>::value)))/2.;
//                 //jump on the face y=bd_dim
//                 auto mean_yplus=eval(in(point+(dof_y+end<1>::value))+in(i+1, point+dof_y))/2.;

//                 //TODO: implement fluxes formula
//                 eval(out(point+dof_x))                 = jump_xplus;
//                 eval(out(point+(dof_x+end<0>::value))) = jump_yminus;
//                 eval(out(point+dof_y))                 = jump_xminus;
//                 eval(out(point+(dof_y+end<1>::value))) = jump_yplus;

//             }
//     }
// };
// // [GD flux]

// [boundary integration]
/**
   This functor computes an integran over a boundary face
*/

// [boundary integration]


int main(){
    //![definitions]
    //dimensions of the problem (in number of elements per dimension)
    auto d1=8;
    auto d2=8;
    auto d3=1;
    //![definitions]
    using namespace enumtype;
    //defining the assembler, based on the Intrepid definitions for the numerics
    using matrix_storage_info_t=storage_info< layout_tt<0,1,2,3,4> , __COUNTER__>;
    using matrix_type=storage_t< matrix_storage_info_t >;
    using fe=reference_element<1, Lagrange, Hexa>;
    using geo_map=reference_element<1, Lagrange, Hexa>;
    using cub=cubature<fe::order, fe::shape>;
    using geo_t = intrepid::geometry<geo_map, cub>;
    using discr_t = intrepid::discretization<fe, cub>;

    // //![boundary]
    // using bd_cub_t = intrepid::boundary_cub<geo_map, 3>;
    // using bd_discr_t = intrepid::boundary_discr<bd_cub_t>;
    // using bd_cub = typename bd_cub_t::bd_cub;
    // bd_cub_t bd_cub_;
    // bd_discr_t bd_discr_(bd_cub_, 0);
    // // compute gradient on the element evaluated on the boundary quadreture points
    // bd_discr_.compute(Intrepid::OPERATOR_GRAD);
    // //![boundary]

    //![instantiation]
    geo_t geo_;
    discr_t fe_;
    fe_.compute(Intrepid::OPERATOR_GRAD);
    //![instantiation]

    using as=assembly< geo_t >;


    //![as_instantiation]
    //constructing the integration tools
    as assembler( geo_, d1, d2, d3);
    //![as_instantiation]

    //![grid]
    //constructing a structured cartesian grid
    for (uint_t i=0; i<d1; i++)
        for (uint_t j=0; j<d2; j++)
            for (uint_t k=0; k<d3; k++)
                for (uint_t point=0; point<fe::basisCardinality; point++)
                {
                    assembler.grid()( i,  j,  k,  point,  0)= (i + geo_.grid()(point, 0));
                    assembler.grid()( i,  j,  k,  point,  1)= (j + geo_.grid()(point, 1));
                    assembler.grid()( i,  j,  k,  point,  2)= (k + geo_.grid()(point, 2));
                    // std::cout<<"grid point("<<m_grid( i,  j,  k,  point,  0) << ", "<< m_grid( i,  j,  k,  point,  1)<<", "<<m_grid( i,  j,  k,  point,  2)<<")"<<std::endl;
                }
    //![grid]

    //![instantiation_stiffness]
    //defining the stiffness matrix: d1xd2xd3 elements
    matrix_storage_info_t meta_(d1,d2,d3,fe::basisCardinality,fe::basisCardinality);
    matrix_type stiffness_(meta_, 0.);
    //![instantiation_stiffness]

    /** defining the computation, i.e. for all elements:
        - computing the jacobian
        - computing its determinant
        - computing the jacobian inverse
        - integrate the stiffness matrix
        - adding the fluxes contribution
    */

    //![placeholders]
    // defining the placeholder for the local gradient of the element boundary face
    typedef arg<as::size, discr_t::grad_storage_t> p_dphi;
    // // defining the placeholder for the local values on the face
    // typedef arg<as::size+4, bd_discr_t::phi_storage_t> p_bd_phi;
    // //output
    typedef arg<as::size+1, matrix_type> p_stiffness;

    // appending the placeholders to the list of placeholders already in place
    // auto domain=assembler.template domain< p_bd_dphi, p_bd_phi, p_flux >( bd_discr_.local_gradient(), bd_discr_.phi(), flux_);
    auto domain=assembler.template domain<p_dphi, p_stiffness>(fe_.local_gradient(), stiffness_);
    //![placeholders]


    // , m_domain(boost::fusion::make_vector(&m_grid, &m_jac, &m_fe_backend.cub_weights(), &m_jac_det, &m_jac_inv, &m_fe_backend.local_gradient(), &m_fe_bac
                                                                                                   // , &m_stiffness, &m_assembled_stiffness
    auto coords=coordinates<axis>({1, 0, 1, d1-1, d1},
                            {1, 0, 1, d2-1, d2});
    coords.value_list[0] = 0;
    coords.value_list[1] = d3-1;

    //![computation]
    auto computation=make_computation<gridtools::BACKEND>(
        make_mss
        (
            execute<forward>(),
            make_esf<functors::update_jac<geo_t> >( as::p_grid_points(), as::p_jac(), p_dphi())
            , make_esf<functors::det<geo_t> >(as::p_jac(), as::p_jac_det())
            , make_esf<functors::inv<geo_t> >(as::p_jac(), as::p_jac_det(), as::p_jac_inv())
            , make_esf<stiffness<fe, cub> >(as::p_jac_det(), as::p_jac_inv(), as::p_weights(), p_stiffness(), p_dphi(), p_dphi())//stiffness
            ), domain, coords);

    computation->ready();
    computation->steady();
    computation->run();
    computation->finalize();
    //![computation]

    return test(assembler, fe_, stiffness_)==true;
}
