#pragma once

namespace functors{

    typedef gridtools::interval<gridtools::level<0,-1>, gridtools::level<1,-1> > x_interval;
    typedef gridtools::interval<gridtools::level<0,-2>, gridtools::level<1,1> > axis;

    template <typename FE, typename Cubature, typename Vec>
    struct advection {
        using fe=FE;
        using cub=Cubature;

        //![accessors]
        using jac_det =accessor<0, range<0,0,0,0> , 4> const;
        using jac_inv =accessor<1, range<0,0,0,0> , 6> const;
        using weights =accessor<2, range<0,0,0,0> , 3> const;
        using dphi    =accessor<3, range<0,0,0,0> , 3> const;
        using psi     =accessor<4, range<0,0,0,0> , 3> const;
        using adv     =accessor<5, range<0,0,0,0> , 5> ;
        using arg_list= boost::mpl::vector<jac_det, jac_inv, weights, adv, dphi, psi> ;
        //![accessors]

        //![Do_advection]
        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_interval) {

            uint_t const num_cub_points=eval.get().get_storage_dims(psi())[1];
            uint_t const basis_cardinality=eval.get().get_storage_dims(psi())[0];

            //quadrature points dimension
            dimension<4>::Index qp;
            //dimension 'i' in the advection matrix
            dimension<5>::Index dimx;
            //dimension 'j' in the advection matrix
            dimension<6>::Index dimy;

            //loop on the basis functions
            for(short_t P_i=0; P_i<basis_cardinality; ++P_i) // current dof
            {
                //loop on the test functions
                for(short_t Q_i=0; Q_i<basis_cardinality; ++Q_i)
                {
                    //loop on the cub points
                    for(short_t q=0; q<num_cub_points; ++q){
                        //inner product of the gradients
                        double inner_product=0.;
                        for(short_t icoor=0; icoor< fe::fe::spaceDim; ++icoor)
                        {
                            inner_product +=
                                eval((jac_inv(qp+q, dimx+0, dimy+icoor)*!dphi(P_i,q,(uint_t)0)+
                                      jac_inv(qp+q, dimx+1, dimy+icoor)*!dphi(P_i,q,(uint_t)1)+
                                      jac_inv(qp+q, dimx+2, dimy+icoor)*!dphi(P_i,q,(uint_t)2))
                                     *
                                     (!psi(Q_i,q,(uint_t)2)*Vec::value[icoor]));
                        }
                        //summing up contributions (times the measure and quad weight)
                        eval(adv(0,0,0,P_i,Q_i)) += inner_product * eval(jac_det(qp+q)*!weights(q,0,0));
                    }
                }
            }
        }
        //![Do_advection]
    };
}//namespace functors
