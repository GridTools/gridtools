namespace functors{
    // [matvec]
    struct matvec {

        using in1=accessor<0, range<> , 4> const;
        using in2=accessor<1, range<> , 5> const;
        using out=accessor<2, range<> , 4> ;
        using arg_list=boost::mpl::vector< in2, in1, out > ;

        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_interval) {
            dimension<4>::Index row;
            dimension<5>::Index col;
            uint_t const cardinality_i=eval.get().get_storage_dims(in2())[3];
            uint_t const cardinality_j=eval.get().get_storage_dims(in2())[4];

            //for all dofs in a boundary face
            for(short_t I=0; I<cardinality_i; I++)
                for(short_t J=0; J<cardinality_j; J++)
                {
                    eval(out(row+I)) = eval(in2(row+I, col+J)*in1(row+J));
                }
        }
    };
    // [matvec]
}; //namespace functors
