#pragma once

namespace gdl{
namespace functors{

    typedef gridtools::interval<gridtools::level<0,-1>, gridtools::level<1,-1> > x_interval;

// [integration]
    struct mass {
        using jac_det =gt::accessor<0, enumtype::in, gt::extent<0,0,0,0> , 4> const;
        using weights =gt::accessor<1, enumtype::in, gt::extent<0,0,0,0> , 3> const;
        using phi     =gt::accessor<2, enumtype::in, gt::extent<0,0,0,0> , 3> const;
        using psi     =gt::accessor<3, enumtype::in, gt::extent<0,0,0,0> , 3> const;
        using mass_t    =gt::accessor<4, enumtype::inout, gt::extent<0,0,0,0> , 5> ;
        using arg_list= boost::mpl::vector<jac_det, weights, phi, psi, mass_t> ;
        using quad=gt::dimension<4>;

        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_interval) {

            uint_t const num_cub_points=eval.get().template get_storage_dims<1>(phi());
            uint_t const basis_cardinality=eval.get().template get_storage_dims<0>(phi());

            quad::Index qp;
            gt::dimension<5>::Index dimx;
            gt::dimension<6>::Index dimy;
            // static int_t dd=fe::hypercube_t::boundary_w_codim<2>::n_points::value;

            //projection of f on a (e.g.) P1 FE space ReferenceFESpace1:
            //loop on quadrature nodes, and on nodes of the P1 element (i,j,k) with i,j,k\in {0,1}
            for(short_t P_i=0; P_i<basis_cardinality; ++P_i) // current dof
            {
                for(short_t Q_i=0; Q_i<basis_cardinality; ++Q_i)
                {//other dofs whose basis function has nonzero support on the element
                    // printf("entry (%d, %d): \n", P_i, Q_i);
                    for(short_t q=0; q<num_cub_points; ++q){
                        assert(eval(jac_det(qp+q)));
                        eval(mass_t((uint_t)0,(uint_t)0,(uint_t)0,(uint_t)P_i,(uint_t)Q_i))  +=
                            eval(!phi(P_i,q,0)*(!psi(Q_i,q,0))*jac_det(qp+q)*!weights(q,0,0));
                        // printf("%f * %f *%f * %f +\n", eval(!phi(P_i,q,0)), eval(!psi(Q_i,q,0)), eval(jac_det(qp+q)), eval(!weights(q,0,0)));
                    }
                    // printf("mass(%d, %d) = %f", P_i, Q_i, eval(mass_t((uint_t)0,(uint_t)0,(uint_t)0,(uint_t)P_i,(uint_t)Q_i)));
                    // printf("\n\n\n\n\n\n");
                    assert( P_i!=Q_i || eval(mass_t((uint_t)0,(uint_t)0,(uint_t)0,(uint_t)P_i,(uint_t)Q_i)));
                }
            }
        }
    };
// [integration]
}//namespace functors
}//namespace gdl
