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
        Dimension<4>::Index qp;
        //dimension 'i' in the stiffness matrix
        Dimension<5>::Index dimx;
        //dimension 'j' in the stiffness matrix
        Dimension<6>::Index dimy;

        //loop on the basis functions
        for(short_t P_i=0; P_i<fe::basisCardinality; ++P_i) // current dof
        {
            //loop on the test functions
            for(short_t Q_i=0; Q_i<fe::basisCardinality; ++Q_i)
            {
                //loop on the cub points
                for(short_t q=0; q<cub::numCubPoints; ++q){
                    //inner product of the gradients
                    double gradients_inner_product=0.;
                    for(short_t icoor=0; icoor< fe::spaceDim; ++icoor)
                    {
                        gradients_inner_product +=
                            eval((jac_inv(qp+q, dimx+0, dimy+icoor)*!dphi(P_i,q,0)+
                                  jac_inv(qp+q, dimx+1, dimy+icoor)*!dphi(P_i,q,1)+
                                  jac_inv(qp+q, dimx+2, dimy+icoor)*!dphi(P_i,q,2))
                                 *
                                 (jac_inv(qp+q, dimx+0, dimy+icoor)*!dphi(Q_i,q,0)+
                                  jac_inv(qp+q, dimx+1, dimy+icoor)*!dphi(Q_i,q,1)+
                                  jac_inv(qp+q, dimx+2, dimy+icoor)*!dphi(Q_i,q,2)));
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

// [DG flux]
/** the following functor will be responsible for computing the DG fluxes (though now only computes the jumps)*/
template <typename FE, typename BdCubature>
struct flux {

    using fe=FE;
    using bd_cub=BdCubature;

    //the accessors are linked to the fields
    using in=accessor<0, range<0,0,0,0> , 5> const;
    using out=accessor<1, range<0,0,0,0> , 5> const;
    using arg_list=boost::mpl::vector<in,out> ;

    // points on the edges
    static const int_t bd_dim=fe::hypercube_t::template boundary_w_dim<1>::n_points::value;
    static const int_t bd_dim_square=bd_dim*bd_dim;
    static const int_t one=1;

    template<ushort_t D>
    struct stride{
        // number of dof on an edge in x direction -1 (to compute the value on an opposite face)
        static const uint_t value=fe::layout_t::template find<D>(bd_dim_square, bd_dim, one);
    };

    template<ushort_t D>
    struct end{
        // number of dof on an edge in x direction -1 (to compute the value on an opposite face)
        static const uint_t value=(stride<D>::value*(bd_dim-1));
    };

    template <typename Evaluation>
    GT_FUNCTION
    static void Do(Evaluation const & eval, x_interval) {
        x::Index i;
        y::Index j;
        z::Index k;
        Dimension<4>::Index point;

        //Hypothesis: the local dofs are ordered according to fe::layout
        array<int, 3> strides={bd_dim*bd_dim, bd_dim, 1};
        //for all dofs in a boundary face
        for(short_t I=0; I<bd_dim; I++)
            for(short_t J=0; J<bd_dim; J++)
            {
                //dof of the basis function
                auto dof_x=(stride<1>::value*I+stride<2>::value*J);
                //jump on the face x=0
                auto jump_xminus=eval(in(point+dof_x)-in(i-1, point+(dof_x+end<0>::value)));
                //jump on the face x=bd_dim
                auto jump_xplus=eval(in(point+(dof_x+end<0>::value))-in(i+1, point+dof_x));
                //dof of the basis function
                auto dof_y=(stride<0>::value*I+stride<2>::value*J);
                //jump on the face y=0
                auto jump_yminus=eval(in(point+dof_y)-in(i-1, point+(dof_y+end<1>::value)));
                //jump on the face y=bd_dim
                auto jump_yplus=eval(in(point+(dof_y+end<1>::value))-in(i+1, point+dof_y));

                //TODO: implement fluxes formula
                eval(out(point+dof_x))                 = jump_xplus;
                eval(out(point+(dof_x+end<0>::value))) = jump_yminus;
                eval(out(point+dof_y))                 = jump_xminus;
                eval(out(point+(dof_y+end<1>::value))) = jump_yplus;

            }
    }
};
// [GD flux]

int main(){
    //![definitions]
    using namespace enumtype;
    //defining the assembler, based on the Intrepid definitions for the numerics
    using matrix_type=storage_t<gridtools::layout_map<0,1,2,3,4> >;
    using fe=reference_element<1, Lagrange, Hexa>;
    using geo_map=reference_element<1, Lagrange, Hexa>;
    using cub=cubature<fe::order, fe::shape>;
    using geo_t = intrepid::geometry<geo_map, cub>;
    using discr_t = intrepid::discretization<fe, cub>;
    using as=assembly< geo_t >;
    //![definitions]

    //dimensions of the problem (in number of elements per dimension)
    auto d1=8;
    auto d2=8;
    auto d3=1;
    //compute the local gradient
    //![instantiation]
    discr_t fe_backend;
    geo_t  geo;
    fe_backend.compute(Intrepid::OPERATOR_GRAD);
    //![instantiation]

    //![as_instantiation]
    //constructing the integration tools
    as assembler(geo,d1,d2,d3);
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
    matrix_type stiffness_(d1,d2,d3,fe::basisCardinality,fe::basisCardinality);

    // initialize with 0
    stiffness_.initialize(0.);
    //![instantiation_stiffness]

    /** defining the computation, i.e. for all elements:
        - computing the jacobian
        - computing its determinant
        - computing the jacobian inverse
        - integrate the stiffness matrix
        - adding the fluxes contribution
    */
    //![boundary]
    using bd=boundary_shape<fe>;
    using bd_cub=cubature<bd::order, bd::shape>;
    using bd_discr_t=intrepid::discretization<bd, bd_cub> ;
    bd_discr_t bd_fe_backend;
    bd_fe_backend.compute(Intrepid::OPERATOR_GRAD);
    //![boundary]

    //![placeholders]
    // defining the placeholder for the stiffness matrix
    typedef arg<as::size+1, matrix_type> p_stiffness;
    // defining the placeholder for the local gradient (it can be different from the one
    // used fro the jacobian, i.e. as::p_dphi)
    typedef arg<as::size+2, discr_t::grad_storage_t> p_dphi;
    // defining the placeholder for the local gradient of the element boundary face
    typedef arg<as::size+3, bd_discr_t::grad_storage_t> p_bd_dphi;

    // appending the placeholders to the list of placeholders already in place
    auto domain=assembler.template domain<p_stiffness, p_dphi, p_bd_dphi>(stiffness_, fe_backend.local_gradient(), bd_fe_backend.local_gradient());
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
            execute<forward>(),
            make_esf<as::update_jac>( as::p_grid_points(), as::p_jac(), as::p_dphi())
            , make_esf<as::det>(as::p_jac(), as::p_jac_det())
            , make_esf<as::inv>(as::p_jac(), as::p_jac_det(), as::p_jac_inv())
            , make_esf<stiffness<fe, cub> >(as::p_jac_det(), as::p_jac_inv(), as::p_weights(), p_stiffness(), p_dphi(), p_dphi())
            // , make_esf<flux<fe, cub > >( p_bd_stiffness(), p_bd_dphi(), p_bd_dphi() )
            ), domain, coords);

    computation->ready();
    computation->steady();
    computation->run();
    computation->finalize();
    //![computation]

    return test(assembler, fe_backend, stiffness_)==true;
}
