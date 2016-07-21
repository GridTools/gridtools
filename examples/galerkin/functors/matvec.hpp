#pragma once


namespace gdl{

    namespace functors{

        typedef gridtools::interval<gridtools::level<0,-1>, gridtools::level<1,-1> > x_interval;

        // [vec_binary_operators]
        template <typename T>
        struct sum_operator {

            inline static T eval(T const & i_a, T const & i_b) { return i_a+i_b; }

        };

        template <typename T>
        struct sub_operator {

            inline static T eval(T const & i_a, T const & i_b) { return i_a-i_b; }

        };

        template <typename T>
        struct mult_operator {

            inline static T eval(T const & i_a, T const & i_b) { return i_a*i_b; }

        };
        // [vec_binary_operators]


        // [vecvec]
        // TODO: these element by element operations could be parallelized avoiding the functor loop
        // Element-wise z = x op y where x, y and z are vectors of the same size
        template <ushort_t Dim, class Operator>
        struct vecvec;

        template <class Operator>
        struct vecvec<4, Operator> {

            using in1=gt::accessor<0, enumtype::in, gt::extent<> , 4> ;
            using in2=gt::accessor<1, enumtype::in, gt::extent<> , 4> ;
            using out=gt::accessor<2, enumtype::inout, gt::extent<> , 4> ;
            using arg_list=boost::mpl::vector< in1, in2, out > ;

            template <typename Evaluation>
            GT_FUNCTION
            static void Do(Evaluation const & eval, x_interval) {
                gt::dimension<4>::Index row;
                uint_t const num_rows=eval.get().template get_storage_dims<3>(in1());

                // Loop over vector elements
                for(uint_t i=0;i<num_rows;++i){
                    eval(out(row+i)) = Operator::eval(eval(in1(row+i)),eval(in2(row+i)));
                }
            }
        }
    };
    // [vecvec]


    // [matvec]
    // left-multiply b = A x
    struct matvec {

        using in2=gt::accessor<0, enumtype::in, gt::extent<> , 5> const;
        using in1=gt::accessor<1, enumtype::in, gt::extent<> , 4> const;
        using out=gt::accessor<2, enumtype::inout, gt::extent<> , 4> ;
        using arg_list=boost::mpl::vector< in2, in1, out > ;

        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_interval) {
            gt::dimension<4>::Index row;
            gt::dimension<5>::Index col;
            uint_t const cardinality_i=eval.template get_storage_dims<3>(in2());
            uint_t const cardinality_j=eval.template get_storage_dims<4>(in2());

            //for all dofs in a boundary face
            for(short_t I=0; I<cardinality_i; I++)
            {
                for(short_t J=0; J<cardinality_j; J++)
                {
                    for(short_t J=0; J<cardinality_j; J++)
                    {
                        eval(out(row+I)) += eval(in2(row+I, col+J)*in1(row+J));
                    }
                }
            }
        }
    };
    // [matvec]

    // [matvec_bd]
    struct matvec_VolxBdxVol {

        using in1=gt::accessor<0, enumtype::in, gt::extent<> , 4> const;
        using in2=gt::accessor<1, enumtype::in, gt::extent<> , 6> const;
        using out=gt::accessor<2, enumtype::inout, gt::extent<> , 4> ;
        using arg_list=boost::mpl::vector< in2, in1, out > ;

        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_interval) {
            gt::dimension<4>::Index row;
            gt::dimension<5>::Index col;
            gt::dimension<6>::Index face;

            uint_t const cardinality_i=eval.template get_storage_dims<3>(in2());
            uint_t const cardinality_j=eval.template get_storage_dims<4>(in2());
            uint_t const faces_=eval.template get_storage_dims<5>(in2());

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

        using in1=gt::accessor<0, enumtype::in, gt::extent<> , 5> const;
        using in2=gt::accessor<1, enumtype::in, gt::extent<> , 6> const;
        using out=gt::accessor<2, enumtype::inout, gt::extent<> , 5> ;
        using arg_list=boost::mpl::vector< in2, in1, out > ;

        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_interval) {
            gt::dimension<4>::Index row;
            gt::dimension<5>::Index col;
            gt::dimension<6>::Index face;

            uint_t const cardinality_i=eval.template get_storage_dims<3>(in2());
            uint_t const cardinality_j=eval.template get_storage_dims<4>(in2());
            uint_t const faces_=eval.template get_storage_dims<5>(in2());

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

    } //namespace functors
} //namespace gdl
