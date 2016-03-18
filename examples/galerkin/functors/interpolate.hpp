namespace gdl{
namespace functors{

    // [counter transformation]
    struct counter_transform {

        using phi   =gt::accessor<0, enumtype::in   , gt::extent<> , 3> const;
        using in   =gt::accessor<1, enumtype::in   , gt::extent<> , 4> const;
        using out   =gt::accessor<2, enumtype::inout, gt::extent<> , 4> ;
        using arg_list=boost::mpl::vector< phi, in, out > ;

        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_interval) {
            gt::dimension<4>::Index dof;

            uint_t const num_cub_points=eval.get().template get_storage_dims<1>(phi());
            uint_t const basis_cardinality=eval.get().template get_storage_dims<0>(phi());

            for(short_t q=0; q<num_cub_points; ++q){
                for(short_t P_i=0; P_i<basis_cardinality; ++P_i) // current dof
                {
                    eval(out((uint_t)0,(uint_t)0,(uint_t)0,(uint_t)q))  +=
                        eval(!phi(P_i,q,0)*in(dof+P_i));
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
            gt::dimension<4>::Index qp;
            gt::dimension<5>::Index dimx;

            uint_t const num_cub_points=eval.get().template get_storage_dims<1>(phi());
            uint_t const basis_cardinality=eval.get().template get_storage_dims<0>(phi());

            for(short_t P_i=0; P_i<basis_cardinality; ++P_i) // current dof
            {
                for(short_t q=0; q<num_cub_points; ++q){
                    assert(eval(jac_det(qp+q)));
                    eval(out((uint_t)0,(uint_t)0,(uint_t)0,(uint_t)P_i))  +=
                        eval(!phi(P_i,q,0)*in(qp+q)*jac_det(qp+q)*!weights(q,0,0));
                }
            }
        }
    };
    // [transformation]


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
            gt::dimension<4>::Index qp;
            gt::dimension<5>::Index dimx;

            uint_t const num_cub_points=eval.get().template get_storage_dims<1>(phi());
            uint_t const basis_cardinality=eval.get().template get_storage_dims<0>(phi());

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
