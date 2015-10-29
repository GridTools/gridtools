/**
\file
*/
#pragma once
#define PEDANTIC_DISABLED
#include "../functors/assembly.hpp"

// [integration]
template <typename FE, typename Cubature>
struct integration {
    using jac_det =accessor<0, range<0,0,0,0> , 4> const;
    using weights =accessor<1, range<0,0,0,0> , 3> const;
    using mass    =accessor<2, range<0,0,0,0> , 5> ;
    using phi    =accessor<3, range<0,0,0,0> , 3> const;
    using psi    =accessor<4, range<0,0,0,0> , 3> const;
    using arg_list= boost::mpl::vector<jac_det, weights, mass, phi, psi> ;
    using quad=dimension<4>;

    using fe=FE;
    using cub=Cubature;

    template <typename Evaluation>
    GT_FUNCTION
    static void Do(Evaluation const & eval, x_interval) {

        quad::Index qp;
        dimension<5>::Index dimx;
        dimension<6>::Index dimy;
        // static int_t dd=fe::hypercube_t::boundary_w_codim<2>::n_points::value;

        //projection of f on a (e.g.) P1 FE space ReferenceFESpace1:
        //loop on quadrature nodes, and on nodes of the P1 element (i,j,k) with i,j,k\in {0,1}
        for(short_t P_i=0; P_i<fe::basisCardinality; ++P_i) // current dof
        {
            for(short_t Q_i=0; Q_i<fe::basisCardinality; ++Q_i)
            {//other dofs whose basis function has nonzero support on the element
                for(short_t q=0; q<cub::numCubPoints(); ++q){
                    eval(mass(0,0,0,P_i,Q_i))  +=
                        eval(!phi(P_i,q,0)*(!psi(Q_i,q,0))*jac_det(qp+q)*!weights(q,0,0));
                }
            }
        }
    }
};
// [integration]

int main(){
    using matrix_storage_info_t=storage_info< layout_tt<3,4> , __COUNTER__>;
    using matrix_type=storage_t< matrix_storage_info_t >;
    using fe=reference_element<1, Lagrange, Hexa>;
    using geo_map=reference_element<1, Lagrange, Hexa>;
    using cub=cubature<fe::order, fe::shape>;
    using geo_t = intrepid::geometry<geo_map, cub>;
    using discr_t = intrepid::discretization<fe, cub>;
    using as=assembly<geo_t>;

    auto d1=8;
    auto d2=8;
    auto d3=1;

    //![instantiation]
    geo_t geo_;
    discr_t fe_;
    fe_.compute(Intrepid::OPERATOR_VALUE);
    //![instantiation]

    as assembler(geo_,d1,d2,d3);

    matrix_storage_info_t meta_(d1,d2,d3,fe::basisCardinality,fe::basisCardinality);
    matrix_type mass_(meta_, 0., "mass");

    typedef arg<as::size, discr_t::basis_function_storage_t> p_phi;
    typedef arg<as::size+1, matrix_type> p_mass;

    auto domain=assembler.template domain<p_phi, p_mass>(fe_.val(), mass_);

    auto coords=coordinates<axis>({1, 0, 1, d1-1, d1},
                            {1, 0, 1, d2-1, d2});
    coords.value_list[0] = 0;
    coords.value_list[1] = d3-1;

    auto computation=make_computation<gridtools::BACKEND>(
        make_mss
        (
            execute<forward>(),
            make_esf<functors::update_jac<geo_t> >( as::p_grid_points(), p_phi(), as::p_jac())
            , make_esf<functors::det<geo_t> >(as::p_jac(), as::p_jac_det())
            , make_esf<integration<fe, cub> >(as::p_jac_det(), as::p_weights(), p_mass(), p_phi(), p_phi())
            // , make_esf<as::assembly_f>(p_mass(), p_assembled_stiffness())
            ), domain, coords);

    computation->ready();
    computation->steady();
    computation->run();
    computation->finalize();
}
