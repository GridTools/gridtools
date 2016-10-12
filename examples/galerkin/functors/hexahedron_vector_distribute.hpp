namespace gdl{
    namespace functors{
    /**
      @class hexahedron mesh assebled vector (single dof indexed object) distribution functor

          After the assemble operation performed by the hexahedron_vector_assemble functor,
          or starting from an already assembled vector, this functor performs the copy of
          the vector values corresponding to shared dof (with adjacent mesh elements) to
          the storages of the adjacent element themselves. Same dof numbering rule described
          for hexahedron_vector_assemble is used (see above).

      @tparam Number of single hexahedron dofs along x direction
      @tparam Number of single hexahedron dofs along y direction
      @tparam Number of single hexahedron dofs along z direction

     */
    // TODO: check todos and comments of previous functor
    // TODO: is this functor the same of uniform?
    template <ushort_t N_DOF0, ushort_t N_DOF1, ushort_t N_DOF2>
    struct hexahedron_vector_distribute {

        using inout=gt::accessor<0, enumtype::inout, gt::extent<> , 4> ;
        using arg_list=boost::mpl::vector<inout> ;

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

                    eval(inout(i-1,dof+indexing.index(I1,J1,indexing.template dim<2>()-1))) =
                            eval(inout(dof+indexing.index(I1,J1,0)));

                    eval(inout(j-1,dof+indexing.index(J1,indexing.template dim<1>()-1,I1))) =
                            eval(inout(dof+indexing.index(J1,0,I1)));

                    eval(inout(k-1,dof+indexing.index(indexing.template dim<0>()-1,I1,J1))) =
                            eval(inout(dof+indexing.index(0,I1,J1)));
                }

            // 2 B
            short_t J1=0;
            for(short_t I1=1; I1<indexing.template dim<0>()-1; I1++)
            {

                eval(inout(i-1,dof+indexing.index(I1,J1,indexing.template dim<2>()-1))) =
                        eval(inout(dof+indexing.index(I1,J1,0)));
                eval(inout(j-1,dof+indexing.index(I1,indexing.template dim<1>()-1,0))) =
                        eval(inout(dof+indexing.index(I1,J1,0)));
                eval(inout(i-1,j-1,dof+indexing.index(I1,indexing.template dim<1>()-1,indexing.template dim<2>()-1))) =
                        eval(inout(dof+indexing.index(I1,J1,0)));


                eval(inout(j-1,dof+indexing.index(J1,indexing.template dim<1>()-1,I1))) =
                        eval(inout(dof+indexing.index(J1,0,I1)));
                eval(inout(k-1,dof+indexing.index(indexing.template dim<0>()-1,0,I1))) =
                        eval(inout(dof+indexing.index(J1,0,I1)));
                eval(inout(j-1,k-1,dof+indexing.index(indexing.template dim<0>()-1,indexing.template dim<1>()-1,I1))) =
                        eval(inout(dof+indexing.index(J1,0,I1)));


                eval(inout(k-1,dof+indexing.index(indexing.template dim<0>()-1,I1,J1))) =
                        eval(inout(dof+indexing.index(0,I1,J1)));
                eval(inout(i-1,dof+indexing.index(0,I1,indexing.template dim<2>()-1))) =
                        eval(inout(dof+indexing.index(0,I1,J1)));
                eval(inout(i-1,k-1,dof+indexing.index(indexing.template dim<0>()-1,I1,indexing.template dim<2>()-1))) =
                        eval(inout(dof+indexing.index(0,I1,J1)));


            }

            // 3 F
            eval(inout(i-1,dof+indexing.index(0,0,indexing.template dim<2>()-1))) =
                eval(inout(dof+0));

            eval(inout(j-1,dof+indexing.index(0,indexing.template dim<1>()-1,0))) =
                eval(inout(dof+0));

            eval(inout(i-1,j-1,dof+indexing.index(0,indexing.template dim<1>()-1,indexing.template dim<2>()-1))) =
                eval(inout(dof+0));

            eval(inout(k-1,dof+indexing.index(indexing.template dim<0>()-1,0,0))) =
                eval(inout(dof+0));

            eval(inout(i-1,k-1,dof+indexing.index(indexing.template dim<0>()-1,0,indexing.template dim<2>()-1))) =
                eval(inout(dof+0));

            eval(inout(j-1,k-1,dof+indexing.index(indexing.template dim<0>()-1,indexing.template dim<1>()-1,0))) =
                eval(inout(dof+0));

            eval(inout(i-1,j-1,k-1,dof+indexing.index(indexing.template dim<0>()-1,indexing.template dim<1>()-1,indexing.template dim<2>()-1))) =
                eval(inout(dof+0));
        }

    };
    } // namespace functors
} // namespace gdl
