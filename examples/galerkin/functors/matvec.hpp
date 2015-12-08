namespace functors{
    // [matvec]


    struct matvec {

        using in1=accessor<0, enumtype::in, extent<> , 4> const;
        using in2=accessor<1, enumtype::in, extent<> , 5> const;
        using out=accessor<2, enumtype::inout, extent<> , 4> ;
        using arg_list=boost::mpl::vector< in2, in1, out > ;

        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_interval) {
            dimension<4>::Index row;
            dimension<5>::Index col;
            uint_t const cardinality_i=eval.get().template get_storage_dims<3>(in2());
            uint_t const cardinality_j=eval.get().template get_storage_dims<4>(in2());

            //for all dofs in a boundary face
            for(short_t I=0; I<cardinality_i; I++)
                for(short_t J=0; J<cardinality_j; J++)
                {
                    eval(out(row+I)) += eval(in2(row+I, col+J)*in1(row+J));
                }
        }
    };
    // [matvec]

    // [matvec_bd]
    struct matvec_VolxBdxVol {

        using in1=accessor<0, enumtype::in, extent<> , 4> const;
        using in2=accessor<1, enumtype::in, extent<> , 6> const;
        using out=accessor<2, enumtype::inout, extent<> , 4> ;
        using arg_list=boost::mpl::vector< in2, in1, out > ;

        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_interval) {
            dimension<4>::Index row;
            dimension<5>::Index col;
            dimension<6>::Index face;

            uint_t const cardinality_i=eval.get().template get_storage_dims<3>(in2());
            uint_t const cardinality_j=eval.get().template get_storage_dims<4>(in2());
            uint_t const faces_=eval.get().template get_storage_dims<5>(in2());

            //for all dofs in a boundary face
            for(short_t I=0; I<cardinality_i; I++)
                for(short_t J=0; J<cardinality_j; J++)
                    for(short_t K=0; K<faces_; K++)
                    {
                        /** TODO: most of the values are 0 in the matrix => should reduce the size. */
                        eval(out(row+I)) += eval(in2(row+I, col+J, face+K)*in1(row+J));
                    }
        }
    };
    // [matvec_bd]

    // [matvec_bd]
    struct matvec_BdxBdxBd {

        using in1=accessor<0, enumtype::in, extent<> , 5> const;
        using in2=accessor<1, enumtype::in, extent<> , 6> const;
        using out=accessor<2, enumtype::inout, extent<> , 5> ;
        using arg_list=boost::mpl::vector< in2, in1, out > ;

        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_interval) {
            dimension<4>::Index row;
            dimension<5>::Index col;
            dimension<6>::Index face;

            uint_t const cardinality_i=eval.get().template get_storage_dims<3>(in2());
            uint_t const cardinality_j=eval.get().template get_storage_dims<4>(in2());
            uint_t const faces_=eval.get().template get_storage_dims<5>(in2());

            //for all dofs in a boundary face
            for(short_t I=0; I<cardinality_i; I++)
                for(short_t J=0; J<cardinality_j; J++)
                    for(short_t K=0; K<faces_; K++)
                    {
                        /** TODO: most of the values are 0 in the matrix => should reduce the size. */
                        eval(out(row+I)) += eval(in2(row+I, col+J, face+K)*in1(row+J));
                    }
        }
    };
    // [matvec_bd]

}; //namespace functors
