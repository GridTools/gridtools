#pragma once

namespace gdl{
namespace functors{

    typedef gridtools::interval<gridtools::level<0,-1>, gridtools::level<1,-1> > x_interval;
    typedef gridtools::interval<gridtools::level<0,-2>, gridtools::level<1,1> > axis;

    template <typename Geo, typename Cubature>
    struct advection {
        using geo_t=Geo;
        using cub=Cubature;

        //![gt::accessors]
        using jac_det =gt::accessor<0, enumtype::in, gt::extent<0,0,0,0> , 4> const;
        using jac_inv =gt::accessor<1, enumtype::in, gt::extent<0,0,0,0> , 6> const;
        using weights =gt::accessor<2, enumtype::in, gt::extent<0,0,0,0> , 3> const;
        using beta    =gt::accessor<3, enumtype::in, gt::extent<0,0,0,0> , 5> const;
        using dphi    =gt::accessor<4, enumtype::in, gt::extent<0,0,0,0> , 3> const;
        using psi     =gt::accessor<5, enumtype::in, gt::extent<0,0,0,0> , 3> const;
        using adv     =gt::accessor<6, enumtype::inout, gt::extent<0,0,0,0> , 5> ;
        using arg_list= boost::mpl::vector< jac_det, jac_inv, weights, beta, dphi, psi, adv > ;
        //![gt::accessors]

        //![Do_advection]
        template <typename Evaluation >
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_interval) {

            uint_t const num_cub_points=eval.template get_storage_dim<1>(psi());
            uint_t const basis_cardinality=eval.template get_storage_dim<0>(psi());

            //quadrature points dimension
            gt::dimension<4> qp;
            //dimension 'i' in the advection matrix
            gt::dimension<5> dimx;
            //dimension 'j' in the advection matrix
            gt::dimension<6> dimy;


            double inner_product=0.;
            //loop on the basis functions
            for(short_t P_i=0; P_i<basis_cardinality; ++P_i) // current dof
            {
                //loop on the test functions
                for(short_t Q_i=0; Q_i<basis_cardinality; ++Q_i)
                {
                    //loop on the cub points
                    for(short_t q=0; q<num_cub_points; ++q){

                        //inner product of the gradients
                        inner_product=0.;
                        for(short_t icoor=0; icoor< 3/*geo_t::fe::space_dim()*/; ++icoor)
                        {

                            //A(k,l)=J_{i,j}*d_i(phi_k)*psi_l*a_j
                            inner_product +=
                                eval(
                                    (
                                        jac_inv(qp+q, dimy+ icoor, dimx+0)
                                        *
                                        !dphi(P_i,q,0)
                                        +
                                        jac_inv(qp+q, dimy+ icoor, dimx+1)
                                        *
                                        !dphi(P_i,q,1)
                                        +
                                        jac_inv(qp+q, dimy+ icoor, dimx+2)
                                        *
                                        !dphi(P_i,q,2)
                                        )
                                    *
                                    beta(0,0,0,P_i,icoor)
                                    );
                        }

                        //summing up contributions (times the measure and quad weight)
                        eval(adv(0,0,0,P_i,Q_i)) -= inner_product*
                            eval(
                                !psi(Q_i,q)
                                *jac_det(qp+q)*!weights(q,0,0));
                    }
                }
            }
        }
    };
}//namespace functors
}//namespace gdl
