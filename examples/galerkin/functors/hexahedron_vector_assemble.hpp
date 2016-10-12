namespace gdl{
    namespace functors{
    /**
      @class hexahedron mesh vector (single dof indexed object) assemble functor
      @tparam Number of single hexahedron dofs along x direction
      @tparam Number of single hexahedron dofs along y direction
      @tparam Number of single hexahedron dofs along z direction

      hypotheses:
      - dof number is the same along each element direction (total number of dofs pre element is n_dofs^3)
      - reference frame axes are defined as follows

                                z
                               /
                                -x
                               |
                               y


      - dofs are ordered in the input matrix according to the following rule

                             4----5
                            /    /|
                           0----1 |
                           |    | 7
                           |    |/
                           2----3

    (not represented internal dofs follow the same rule)

      - each hexahedron is responsible for the assemble of a set of contributions related to the dof shared with the
    adjacent hexaedrons in negative x,y and z direction (each hexahedron takes contribution from 7 adjacent hexahedrons).
    Particularly, considering a single face the dofs are grouped as follows

                        F--B--B--B--G           ------x
                        |           |           |
                        C--A--A--A--E           |
                        |           |           y
                        C--A--A--A--E
                        |           |
                        C--A--A--A--E
                        |           |
                        H--D--D--D--I

    The assemble (gathering) is performed for heach hexahedron on the following dof groups :

    ------------------------------------------------------------------------------
        Loop_number |   Group           |   Number_contr
    ------------------------------------------------------------------------------
        1           |   A               |   2
        2           |   B+C             |   4
        3           |   F               |   8
    ------------------------------------------------------------------------------

    The same calculation is performed for the faces on xz and xy planes.


    As it can be seen some contributions are missing (e.g. E): those elements are included in the calculations of the adjacent
    (in positive x/y/z direction) hexahedron.

     */
    // TODO: check todos and comments of previous functor
    template <ushort_t N_DOF0, ushort_t N_DOF1, ushort_t N_DOF2>
    struct hexahedron_vector_assemble {

        using in=gt::accessor<0, enumtype::in, gt::extent<> , 4> ;
        using out=gt::accessor<1, enumtype::inout, gt::extent<> , 4> ;
        using arg_list=boost::mpl::vector<in, out> ;

        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_interval) {
            gt::dimension<1> k;
            gt::dimension<2> j;
            gt::dimension<3> i;
            gt::dimension<4> dof;

            constexpr gt::meta_storage_base<static_int<__COUNTER__>,gt::layout_map<2,1,0>,false> indexing{N_DOF0,N_DOF1,N_DOF2};

            // 1 A
            for(short_t I1=1; I1<indexing.template dim<0>()-1; I1++)
                for(short_t J1=1; J1<indexing.template dim<1>()-1; J1++)
                {

                    eval(out(dof+indexing.index(I1,J1,0))) +=
                            eval(in(i-1,dof+indexing.index(I1,J1,indexing.template dim<2>()-1)));

                    eval(out(dof+indexing.index(J1,0,I1))) +=
                            eval(in(j-1,dof+indexing.index(J1,indexing.template dim<1>()-1,I1)));

                    eval(out(dof+indexing.index(0,I1,J1))) +=
                            eval(in(k-1,dof+indexing.index(indexing.template dim<0>()-1,I1,J1)));

                }

            // 2 B
            short_t J1=0;
            for(short_t I1=1; I1<indexing.template dim<0>()-1; I1++)
            {

                eval(out(dof+indexing.index(I1,J1,0))) +=
                        eval(in(i-1,dof+indexing.index(I1,J1,indexing.template dim<2>()-1))) +
                        eval(in(j-1,dof+indexing.index(I1,indexing.template dim<1>()-1,0))) +
                        eval(in(i-1,j-1,dof+indexing.index(I1,indexing.template dim<1>()-1,indexing.template dim<2>()-1)));


                eval(out(dof+indexing.index(J1,0,I1))) +=
                        eval(in(j-1,dof+indexing.index(J1,indexing.template dim<1>()-1,I1))) +
                        eval(in(k-1,dof+indexing.index(indexing.template dim<0>()-1,0,I1))) +
                        eval(in(j-1,k-1,dof+indexing.index(indexing.template dim<0>()-1,indexing.template dim<1>()-1,I1)));


                eval(out(dof+indexing.index(0,I1,J1))) +=
                        eval(in(k-1,dof+indexing.index(indexing.template dim<0>()-1,I1,J1))) +
                        eval(in(i-1,dof+indexing.index(0,I1,indexing.template dim<2>()-1))) +
                        eval(in(i-1,k-1,dof+indexing.index(indexing.template dim<0>()-1,I1,indexing.template dim<2>()-1)));


            }

            // 3 F

            eval(out(dof+0)) +=
                eval(in(i-1,dof+indexing.index(0,0,indexing.template dim<2>()-1))) +
                eval(in(j-1,dof+indexing.index(0,indexing.template dim<1>()-1,0))) +
                eval(in(i-1,j-1,dof+indexing.index(0,indexing.template dim<1>()-1,indexing.template dim<2>()-1))) +
                eval(in(k-1,dof+indexing.index(indexing.template dim<0>()-1,0,0))) +
                eval(in(i-1,k-1,dof+indexing.index(indexing.template dim<0>()-1,0,indexing.template dim<2>()-1))) +
                eval(in(j-1,k-1,dof+indexing.index(indexing.template dim<0>()-1,indexing.template dim<1>()-1,0))) +
                eval(in(i-1,j-1,k-1,dof+indexing.index(indexing.template dim<0>()-1,indexing.template dim<1>()-1,indexing.template dim<2>()-1)));

        }

    };
    }// namespace functors
}// namespace gdl
