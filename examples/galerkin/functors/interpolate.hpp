namespace gdl{
namespace functors{

    // [counter transformation]
    struct evaluate {

        using phi   =gt::accessor<0, enumtype::in   , gt::extent<> , 3> const;
        using in    =gt::accessor<1, enumtype::in   , gt::extent<> , 4> const;
        using weights  =gt::accessor<2, enumtype::inout, gt::extent<> , 3> ;
        using jac_det  =gt::accessor<3, enumtype::inout, gt::extent<> , 4> ;
        using out   =gt::accessor<4, enumtype::inout, gt::extent<> , 4> ;

        using arg_list=boost::mpl::vector< phi, in, weights, jac_det, out > ;

        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_interval) {
            gt::dimension<4> dof;

            uint_t const num_cub_points=eval.template get_storage_dim<1>(phi());
            uint_t const basis_cardinality=eval.template get_storage_dim<0>(phi());
            for(short_t q=0; q<num_cub_points; ++q){
                eval(out(0,0,0,q))=0.;
            }

            for(short_t P_i=0; P_i<basis_cardinality; ++P_i) // current dof
            {
                double normalization=0.;
                for(short_t q=0; q<num_cub_points; ++q){
                    normalization  +=
                        eval(!phi(P_i,q,0)*!phi(P_i,q,0)*!weights(q,0,0) *jac_det(0,0,0,q));
                }
                for(short_t q=0; q<num_cub_points; ++q){
                    eval(out(0,0,0,q))  +=
                        eval( !phi(P_i,q,0)*
                              in(dof+P_i))/std::sqrt(normalization);
                }
            }
        }
    };
    // [counter transformation]

    // [transformation]
    struct transform {

        using jac_det=gt::accessor<0, enumtype::in   , gt::extent<> , 4> const;
        using weights   =gt::accessor<1, enumtype::inout, gt::extent<> , 3> ;
        using phi   =gt::accessor<2, enumtype::in   , gt::extent<> , 3> const;
        using in   =gt::accessor<3, enumtype::in   , gt::extent<> , 4> const;
        using out   =gt::accessor<4, enumtype::inout, gt::extent<> , 4> ;
        using arg_list=boost::mpl::vector< jac_det, weights, phi, in, out > ;

        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_interval) {
            gt::dimension<4> qp;
            gt::dimension<5> dimx;

            uint_t const num_cub_points=eval.template get_storage_dim<1>(phi());
            uint_t const basis_cardinality=eval.template get_storage_dim<0>(phi());

            for(short_t P_i=0; P_i<basis_cardinality; ++P_i) // current dof
            {
                double normalization = 0.;
                for(short_t q=0; q<num_cub_points; ++q){
                    normalization  +=
                        eval(!phi(P_i,q,0)*!phi(P_i,q,0)*!weights(q,0,0) *jac_det(qp+q)
                                      );
                }

                assert(normalization>0);

                eval(out((uint_t)0,(uint_t)0,(uint_t)0,(uint_t)P_i))=0.;
                for(short_t q=0; q<num_cub_points; ++q){
                    assert(eval(jac_det(qp+q)));
                    eval(out((uint_t)0,(uint_t)0,(uint_t)0,(uint_t)P_i))  +=
                        eval(!phi(P_i,q,0)*in(qp+q)/std::sqrt(normalization)
                             *!weights(q,0,0) * jac_det(qp+q)
                            );

                }

            }
        }
    };
    // [transformation]


    // just code repetition
    struct transform_vec {

        using jac_det=gt::accessor<0, enumtype::in   , gt::extent<> , 4> const;
        using weights   =gt::accessor<1, enumtype::in, gt::extent<> , 3> ;
        using phi   =gt::accessor<2, enumtype::in   , gt::extent<> , 3> const;
        using in   =gt::accessor<3, enumtype::in   , gt::extent<> , 5> const;
        using out   =gt::accessor<4, enumtype::inout, gt::extent<> , 5> ;
        using arg_list=boost::mpl::vector< jac_det, weights, phi, in, out > ;

        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_interval) {
            gt::dimension<4> qp;
            gt::dimension<5> dimx;

            uint_t const num_cub_points=eval.template get_storage_dim<1>(phi());
            uint_t const basis_cardinality=eval.template get_storage_dim<0>(phi());
            uint_t const space_dim=eval.template get_storage_dim<4>(in());

            for(short_t d=0; d<space_dim; ++d) // space dimension
            {
                for(short_t P_i=0; P_i<basis_cardinality; ++P_i) // current dof
                {
                    double normalization = 0.;
                    for(short_t q=0; q<num_cub_points; ++q){
                        normalization  +=
                            eval(!phi(P_i,q,0)*!phi(P_i,q,0)*!weights(q,0,0) *jac_det(qp+q));
                    }

                    eval(out((uint_t)0,(uint_t)0,(uint_t)0,(uint_t)P_i))=0.;
                    for(short_t q=0; q<num_cub_points; ++q){
                        assert(eval(jac_det(qp+q)));
                        eval(out((uint_t)0,(uint_t)0,(uint_t)0,(uint_t)P_i, d))  +=
                            eval(!phi(P_i,q,0)*in(qp+q, dimx+d)/std::sqrt(normalization)
                                 *!weights(q,0,0) * jac_det(qp+q)
                                );
                    }
                }
            }
        }
    };


    // [transformation]
    template <ushort_t Face>
    struct bd_transform {

        using jac_det=gt::accessor<0, enumtype::in   , gt::extent<> , 5> const;
        using weights   =gt::accessor<1, enumtype::inout, gt::extent<> , 3> ;
        using phi   =gt::accessor<2, enumtype::in   , gt::extent<> , 3> const;
        using in   =gt::accessor<3, enumtype::in   , gt::extent<> , 5> const;
        using out   =gt::accessor<4, enumtype::inout, gt::extent<> , 5> ;
        using arg_list=boost::mpl::vector< jac_det, weights, phi, in, out > ;

        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_interval) {
            gt::dimension<4> qp;
            gt::dimension<5> dimx;

            uint_t const num_cub_points=eval.template get_storage_dim<1>(phi());
            uint_t const basis_cardinality=eval.template get_storage_dim<0>(phi());

            for(short_t P_i=0; P_i<basis_cardinality; ++P_i) // current dof
            {
                for(short_t q=0; q<num_cub_points; ++q){
                    assert(eval(jac_det(qp+q)));
                    eval(out(0,0,0,P_i,Face))  +=
                        eval(!phi(P_i,q,0)*in(0,0,0,qp+q,Face)*jac_det(0,0,0,qp+q,Face)*!weights(q,0,0));
                }
            }
        }
    };
    // [transformation]

}//namespace functors
}//namespace gdl
