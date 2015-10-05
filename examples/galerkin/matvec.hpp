namespace functors{
    // [matvec]
    template<typename Geometry>
    struct matvec {

        using geo_map=typename Geometry::geo_map;

        using in1=accessor<0, range<> , 4> const;
        using in2=accessor<1, range<> , 5> const;
        using out=accessor<2, range<> , 4> ;
        using arg_list=boost::mpl::vector<in1, in2, out> ;

        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_interval) {
            dimension<4>::Index row;
            dimension<5>::Index col;
            uint_t const cardinality_i=eval.get().get_storage_dims(in1())[3];
            uint_t const cardinality_j=eval.get().get_storage_dims(in1())[4];

#ifndef __CUDACC__
            assert(cardinality_i==fe::basisCardinality);
            assert(cardinality_j==fe::basisCardinality);
#endif

            //for all dofs in a boundary face
            for(short_t I=0; I<cardinality_i; I++)
                for(short_t J=0; J<cardinality_j; J++)
                {
                    eval(out(row+I)) = eval(in1(row+I, col+J)*in2(row+J));
                }
        }
    };
    // [matvec]
}; //namespace functors
