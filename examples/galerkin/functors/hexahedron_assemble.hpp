namespace gdl{
    namespace functors{
    /**
      @class hexahedron mesh matrix (dof pair indexed object) assemble functor
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

      - each hexahedron is responsible for the assemble of a set of contributions related to the dof pairs shared with the
	adjacent hexaedrons in negative x,y and z direction (each hexahedron takes contribution from 7 adjacent hexahedrons).
	Particularly, considering a single face the dofs are grouped as follows

						F--B--B--B--G			------x
						|	        |			|
						C--A--A--A--E			|
						|	        |			y
						C--A--A--A--E
						|	        |
						C--A--A--A--E
						|	        |
						H--D--D--D--I

	The assemble (gathering) is performed for heach hexahedron on the following dof group pairs (and their symmetric pair):

	------------------------------------------------------------------------------
		Loop_number	|	Group_pair		|	Number_contr
	------------------------------------------------------------------------------
		1 		    |	(A,A)			|	2
		2		    |	(A,F+B+G)		|	2
		3		    |	(A,H+D+I)		|	2
		4		    |	(C,B+A+D+G+E+I)	|	2
		5		    |	(B,H+D+I)		|	2
		6		    |	(E,B+A+D)		|	2
		7		    |	(F,D+I)			|	2
		8		    |	(G,H+D)			|	2
		9		    |	(F,E)			|	2
		10		    |	(H,E)			|	2
		11		    |	(F,B+G)			|	4
		12		    |	(G,B)			|	4
		13		    |	(B,B)			|	4
	------------------------------------------------------------------------------

	The same calculation is performed for the faces on xz and xy planes. Moreover, a final "corner" dof pair is included in
	the calculation of the considered hexahedron, namely the (F,F) pair.


	As it can be seen some contributions are missing (e.g. (G,E)): those pairs are included in the calculations of the adjacent
	(in positive x/y/z direction) hexahedron.

     */
    template <ushort_t N_DOF0, ushort_t N_DOF1, ushort_t N_DOF2>
    struct hexahedron_assemble {

        // TODO: matrix symmetry hypothesis not required here (but we know that matrix for this functor are symmetric)
        // TODO: extend to non isotropic dof distribution case
        // TODO: expression within loops are always the same, avoid code duplication
        using in2=gt::accessor<0, enumtype::in, gt::extent<> , 5>;
        using out=gt::accessor<1, enumtype::inout, gt::extent<> , 5> ;
        using arg_list=boost::mpl::vector<in2, out> ;

        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_interval) {
            gt::dimension<1> k;
            gt::dimension<2> j;
            gt::dimension<3> i;
            gt::dimension<4> dof1;
            gt::dimension<5> dof2;

            constexpr gt::meta_storage_base<static_int<__COUNTER__>,gt::layout_map<2,1,0>,false> indexing{N_DOF0,N_DOF1,N_DOF2};

            // 1 (A,A)
            for(short_t I1=1; I1<indexing.template dim<0>()-1; I1++)
                for(short_t J1=1; J1<indexing.template dim<1>()-1; J1++)
                    for(short_t I2=1; I2<indexing.template dim<0>()-1; I2++)
                        for(short_t J2=1; J2<indexing.template dim<1>()-1; J2++)
                        {

                            eval(out(dof1+indexing.index(I1,J1,0),dof2+indexing.index(I2,J2,0))) +=
                                    eval(in2(i-1,dof1+indexing.index(I1,J1,indexing.template dim<2>()-1),dof2+indexing.index(I2,J2,indexing.template dim<2>()-1)));

                            eval(out(dof1+indexing.index(J1,0,I1),dof2+indexing.index(J2,0,I2))) +=
                                    eval(in2(j-1,dof1+indexing.index(J1,indexing.template dim<1>()-1,I1),dof2+indexing.index(J2,indexing.template dim<1>()-1,I2)));

                            eval(out(dof1+indexing.index(0,I1,J1),dof2+indexing.index(0,I2,J2))) +=
                                    eval(in2(k-1,dof1+indexing.index(indexing.template dim<0>()-1,I1,J1),dof2+indexing.index(indexing.template dim<0>()-1,I2,J2)));


                        }

            // 2 (A,F+B+G)+(F+B+G,A)
            short_t J2=0;
            for(short_t I1=1; I1<indexing.template dim<0>()-1; I1++)
                for(short_t J1=1; J1<indexing.template dim<1>()-1; J1++)
                    for(short_t I2=0; I2<indexing.template dim<0>(); I2++)
                    {

                        eval(out(dof1+indexing.index(I1,J1,0),dof2+indexing.index(I2,J2,0))) +=
                                eval(in2(i-1,dof1+indexing.index(I1,J1,indexing.template dim<2>()-1),dof2+indexing.index(I2,J2,indexing.template dim<2>()-1)));
                        eval(out(dof1+indexing.index(I2,J2,0),dof2+indexing.index(I1,J1,0))) +=
                                eval(in2(i-1,dof1+indexing.index(I2,J2,indexing.template dim<2>()-1),dof2+indexing.index(I1,J1,indexing.template dim<2>()-1)));

                        eval(out(dof1+indexing.index(J1,0,I1),dof2+indexing.index(J2,0,I2))) +=
                                eval(in2(j-1,dof1+indexing.index(J1,indexing.template dim<1>()-1,I1),dof2+indexing.index(J2,indexing.template dim<1>()-1,I2)));
                        eval(out(dof1+indexing.index(J2,0,I2),dof2+indexing.index(J1,0,I1))) +=
                                eval(in2(j-1,dof1+indexing.index(J2,indexing.template dim<1>()-1,I2),dof2+indexing.index(J1,indexing.template dim<1>()-1,I1)));

                        eval(out(dof1+indexing.index(0,I1,J1),dof2+indexing.index(0,I2,J2))) +=
                                eval(in2(k-1,dof1+indexing.index(indexing.template dim<0>()-1,I1,J1),dof2+indexing.index(indexing.template dim<0>()-1,I2,J2)));
                        eval(out(dof1+indexing.index(0,I2,J2),dof2+indexing.index(0,I1,J1))) +=
                                eval(in2(k-1,dof1+indexing.index(indexing.template dim<0>()-1,I2,J2),dof2+indexing.index(indexing.template dim<0>()-1,I1,J1)));

                    }

            // 3 (A,H+D+I)+(H+D+I,A)
            J2=indexing.template dim<1>()-1;
            for(short_t I1=1; I1<indexing.template dim<0>()-1; I1++)
                for(short_t J1=1; J1<indexing.template dim<1>()-1; J1++)
                    for(short_t I2=0; I2<indexing.template dim<0>(); I2++)
                    {

                        eval(out(dof1+indexing.index(I1,J1,0),dof2+indexing.index(I2,J2,0))) +=
                                eval(in2(i-1,dof1+indexing.index(I1,J1,indexing.template dim<2>()-1),dof2+indexing.index(I2,J2,indexing.template dim<2>()-1)));
                        eval(out(dof1+indexing.index(I2,J2,0),dof2+indexing.index(I1,J1,0))) +=
                                eval(in2(i-1,dof1+indexing.index(I2,J2,indexing.template dim<2>()-1),dof2+indexing.index(I1,J1,indexing.template dim<2>()-1)));

                        eval(out(dof1+indexing.index(J1,0,I1),dof2+indexing.index(J2,0,I2))) +=
                                eval(in2(j-1,dof1+indexing.index(J1,indexing.template dim<1>()-1,I1),dof2+indexing.index(J2,indexing.template dim<1>()-1,I2)));
                        eval(out(dof1+indexing.index(J2,0,I2),dof2+indexing.index(J1,0,I1))) +=
                                eval(in2(j-1,dof1+indexing.index(J2,indexing.template dim<1>()-1,I2),dof2+indexing.index(J1,indexing.template dim<1>()-1,I1)));

                        eval(out(dof1+indexing.index(0,I1,J1),dof2+indexing.index(0,I2,J2))) +=
                                eval(in2(k-1,dof1+indexing.index(indexing.template dim<0>()-1,I1,J1),dof2+indexing.index(indexing.template dim<0>()-1,I2,J2)));
                        eval(out(dof1+indexing.index(0,I2,J2),dof2+indexing.index(0,I1,J1))) +=
                                eval(in2(k-1,dof1+indexing.index(indexing.template dim<0>()-1,I2,J2),dof2+indexing.index(indexing.template dim<0>()-1,I1,J1)));

                    }

            // 4 (C,B+A+D+G+E+I)+(B+A+D+G+E+I,C)
            short_t I1=0;
            for(short_t J1=1; J1<indexing.template dim<1>()-1; J1++)
                for(short_t I2=1; I2<indexing.template dim<0>(); I2++)
                    for(short_t J2=0; J2<indexing.template dim<1>(); J2++)
                    {

                        eval(out(dof1+indexing.index(I1,J1,0),dof2+indexing.index(I2,J2,0))) +=
                                eval(in2(i-1,dof1+indexing.index(I1,J1,indexing.template dim<2>()-1),dof2+indexing.index(I2,J2,indexing.template dim<2>()-1)));
                        eval(out(dof1+indexing.index(I2,J2,0),dof2+indexing.index(I1,J1,0))) +=
                                eval(in2(i-1,dof1+indexing.index(I2,J2,indexing.template dim<2>()-1),dof2+indexing.index(I1,J1,indexing.template dim<2>()-1)));

                        eval(out(dof1+indexing.index(J1,0,I1),dof2+indexing.index(J2,0,I2))) +=
                                eval(in2(j-1,dof1+indexing.index(J1,indexing.template dim<1>()-1,I1),dof2+indexing.index(J2,indexing.template dim<1>()-1,I2)));
                        eval(out(dof1+indexing.index(J2,0,I2),dof2+indexing.index(J1,0,I1))) +=
                                eval(in2(j-1,dof1+indexing.index(J2,indexing.template dim<1>()-1,I2),dof2+indexing.index(J1,indexing.template dim<1>()-1,I1)));

                        eval(out(dof1+indexing.index(0,I1,J1),dof2+indexing.index(0,I2,J2))) +=
                                eval(in2(k-1,dof1+indexing.index(indexing.template dim<0>()-1,I1,J1),dof2+indexing.index(indexing.template dim<0>()-1,I2,J2)));
                        eval(out(dof1+indexing.index(0,I2,J2),dof2+indexing.index(0,I1,J1))) +=
                                eval(in2(k-1,dof1+indexing.index(indexing.template dim<0>()-1,I2,J2),dof2+indexing.index(indexing.template dim<0>()-1,I1,J1)));

                    }

            // 5 (B,H+D+I)+(H+D+I,B)
            short_t J1=0;
            J2=indexing.template dim<1>() - 1;
            for(short_t I1=1; I1<indexing.template dim<0>()-1; I1++)
                for(short_t I2=0; I2<indexing.template dim<0>(); I2++)
                {

                    eval(out(dof1+indexing.index(I1,J1,0),dof2+indexing.index(I2,J2,0))) +=
                            eval(in2(i-1,dof1+indexing.index(I1,J1,indexing.template dim<2>()-1),dof2+indexing.index(I2,J2,indexing.template dim<2>()-1)));
                    eval(out(dof1+indexing.index(I2,J2,0),dof2+indexing.index(I1,J1,0))) +=
                            eval(in2(i-1,dof1+indexing.index(I2,J2,indexing.template dim<2>()-1),dof2+indexing.index(I1,J1,indexing.template dim<2>()-1)));

                    eval(out(dof1+indexing.index(J1,0,I1),dof2+indexing.index(J2,0,I2))) +=
                            eval(in2(j-1,dof1+indexing.index(J1,indexing.template dim<1>()-1,I1),dof2+indexing.index(J2,indexing.template dim<1>()-1,I2)));
                    eval(out(dof1+indexing.index(J2,0,I2),dof2+indexing.index(J1,0,I1))) +=
                            eval(in2(j-1,dof1+indexing.index(J2,indexing.template dim<1>()-1,I2),dof2+indexing.index(J1,indexing.template dim<1>()-1,I1)));

                    eval(out(dof1+indexing.index(0,I1,J1),dof2+indexing.index(0,I2,J2))) +=
                            eval(in2(k-1,dof1+indexing.index(indexing.template dim<0>()-1,I1,J1),dof2+indexing.index(indexing.template dim<0>()-1,I2,J2)));
                    eval(out(dof1+indexing.index(0,I2,J2),dof2+indexing.index(0,I1,J1))) +=
                            eval(in2(k-1,dof1+indexing.index(indexing.template dim<0>()-1,I2,J2),dof2+indexing.index(indexing.template dim<0>()-1,I1,J1)));

                }

            // 6 (E,B+A+D)+(B+A+D,E)
            I1=indexing.template dim<0>()-1;
            for(short_t J1=1; J1<indexing.template dim<1>()-1; J1++)
                for(short_t I2=1; I2<indexing.template dim<0>()-1; I2++)
                    for(short_t J2=0; J2<indexing.template dim<1>(); J2++)
                    {

                        eval(out(dof1+indexing.index(I1,J1,0),dof2+indexing.index(I2,J2,0))) +=
                                eval(in2(i-1,dof1+indexing.index(I1,J1,indexing.template dim<2>()-1),dof2+indexing.index(I2,J2,indexing.template dim<2>()-1)));
                        eval(out(dof1+indexing.index(I2,J2,0),dof2+indexing.index(I1,J1,0))) +=
                                eval(in2(i-1,dof1+indexing.index(I2,J2,indexing.template dim<2>()-1),dof2+indexing.index(I1,J1,indexing.template dim<2>()-1)));

                        eval(out(dof1+indexing.index(J1,0,I1),dof2+indexing.index(J2,0,I2))) +=
                                eval(in2(j-1,dof1+indexing.index(J1,indexing.template dim<1>()-1,I1),dof2+indexing.index(J2,indexing.template dim<1>()-1,I2)));
                        eval(out(dof1+indexing.index(J2,0,I2),dof2+indexing.index(J1,0,I1))) +=
                                eval(in2(j-1,dof1+indexing.index(J2,indexing.template dim<1>()-1,I2),dof2+indexing.index(J1,indexing.template dim<1>()-1,I1)));

                        eval(out(dof1+indexing.index(0,I1,J1),dof2+indexing.index(0,I2,J2))) +=
                                eval(in2(k-1,dof1+indexing.index(indexing.template dim<0>()-1,I1,J1),dof2+indexing.index(indexing.template dim<0>()-1,I2,J2)));
                        eval(out(dof1+indexing.index(0,I2,J2),dof2+indexing.index(0,I1,J1))) +=
                                eval(in2(k-1,dof1+indexing.index(indexing.template dim<0>()-1,I2,J2),dof2+indexing.index(indexing.template dim<0>()-1,I1,J1)));
                    }



            // 7 (F,D+I)+(D+I,F)
            I1=0;
            J1=0;
            J2=indexing.template dim<1>() - 1;
            for(short_t I2=1; I2<indexing.template dim<0>(); I2++)
            {

                eval(out(dof1+indexing.index(I1,J1,0),dof2+indexing.index(I2,J2,0))) +=
                        eval(in2(i-1,dof1+indexing.index(I1,J1,indexing.template dim<2>()-1),dof2+indexing.index(I2,J2,indexing.template dim<2>()-1)));
                eval(out(dof1+indexing.index(I2,J2,0),dof2+indexing.index(I1,J1,0))) +=
                        eval(in2(i-1,dof1+indexing.index(I2,J2,indexing.template dim<2>()-1),dof2+indexing.index(I1,J1,indexing.template dim<2>()-1)));

                eval(out(dof1+indexing.index(J1,0,I1),dof2+indexing.index(J2,0,I2))) +=
                        eval(in2(j-1,dof1+indexing.index(J1,indexing.template dim<1>()-1,I1),dof2+indexing.index(J2,indexing.template dim<1>()-1,I2)));
                eval(out(dof1+indexing.index(J2,0,I2),dof2+indexing.index(J1,0,I1))) +=
                        eval(in2(j-1,dof1+indexing.index(J2,indexing.template dim<1>()-1,I2),dof2+indexing.index(J1,indexing.template dim<1>()-1,I1)));

                eval(out(dof1+indexing.index(0,I1,J1),dof2+indexing.index(0,I2,J2))) +=
                        eval(in2(k-1,dof1+indexing.index(indexing.template dim<0>()-1,I1,J1),dof2+indexing.index(indexing.template dim<0>()-1,I2,J2)));
                eval(out(dof1+indexing.index(0,I2,J2),dof2+indexing.index(0,I1,J1))) +=
                        eval(in2(k-1,dof1+indexing.index(indexing.template dim<0>()-1,I2,J2),dof2+indexing.index(indexing.template dim<0>()-1,I1,J1)));

            }

            // 8 (G,H+D)+(H+D,G)
            I1=indexing.template dim<1>() - 1;
            J1=0;
            J2=indexing.template dim<1>() - 1;
            for(short_t I2=0; I2<indexing.template dim<0>()-1; I2++)
            {

                eval(out(dof1+indexing.index(I1,J1,0),dof2+indexing.index(I2,J2,0))) +=
                        eval(in2(i-1,dof1+indexing.index(I1,J1,indexing.template dim<2>()-1),dof2+indexing.index(I2,J2,indexing.template dim<2>()-1)));
                eval(out(dof1+indexing.index(I2,J2,0),dof2+indexing.index(I1,J1,0))) +=
                        eval(in2(i-1,dof1+indexing.index(I2,J2,indexing.template dim<2>()-1),dof2+indexing.index(I1,J1,indexing.template dim<2>()-1)));

                eval(out(dof1+indexing.index(J1,0,I1),dof2+indexing.index(J2,0,I2))) +=
                        eval(in2(j-1,dof1+indexing.index(J1,indexing.template dim<1>()-1,I1),dof2+indexing.index(J2,indexing.template dim<1>()-1,I2)));
                eval(out(dof1+indexing.index(J2,0,I2),dof2+indexing.index(J1,0,I1))) +=
                        eval(in2(j-1,dof1+indexing.index(J2,indexing.template dim<1>()-1,I2),dof2+indexing.index(J1,indexing.template dim<1>()-1,I1)));

                eval(out(dof1+indexing.index(0,I1,J1),dof2+indexing.index(0,I2,J2))) +=
                        eval(in2(k-1,dof1+indexing.index(indexing.template dim<0>()-1,I1,J1),dof2+indexing.index(indexing.template dim<0>()-1,I2,J2)));
                eval(out(dof1+indexing.index(0,I2,J2),dof2+indexing.index(0,I1,J1))) +=
                        eval(in2(k-1,dof1+indexing.index(indexing.template dim<0>()-1,I2,J2),dof2+indexing.index(indexing.template dim<0>()-1,I1,J1)));

            }

            // 9 (F,E)+(E,F)
            I1=0;
            J1=0;
            short_t I2=indexing.template dim<0>()-1;
            for(short_t J2=1; J2<indexing.template dim<1>()-1; J2++)
            {

                eval(out(dof1+indexing.index(I1,J1,0),dof2+indexing.index(I2,J2,0))) +=
                        eval(in2(i-1,dof1+indexing.index(I1,J1,indexing.template dim<2>()-1),dof2+indexing.index(I2,J2,indexing.template dim<2>()-1)));
                eval(out(dof1+indexing.index(I2,J2,0),dof2+indexing.index(I1,J1,0))) +=
                        eval(in2(i-1,dof1+indexing.index(I2,J2,indexing.template dim<2>()-1),dof2+indexing.index(I1,J1,indexing.template dim<2>()-1)));

                eval(out(dof1+indexing.index(J1,0,I1),dof2+indexing.index(J2,0,I2))) +=
                        eval(in2(j-1,dof1+indexing.index(J1,indexing.template dim<1>()-1,I1),dof2+indexing.index(J2,indexing.template dim<1>()-1,I2)));
                eval(out(dof1+indexing.index(J2,0,I2),dof2+indexing.index(J1,0,I1))) +=
                        eval(in2(j-1,dof1+indexing.index(J2,indexing.template dim<1>()-1,I2),dof2+indexing.index(J1,indexing.template dim<1>()-1,I1)));

                eval(out(dof1+indexing.index(0,I1,J1),dof2+indexing.index(0,I2,J2))) +=
                        eval(in2(k-1,dof1+indexing.index(indexing.template dim<0>()-1,I1,J1),dof2+indexing.index(indexing.template dim<0>()-1,I2,J2)));
                eval(out(dof1+indexing.index(0,I2,J2),dof2+indexing.index(0,I1,J1))) +=
                        eval(in2(k-1,dof1+indexing.index(indexing.template dim<0>()-1,I2,J2),dof2+indexing.index(indexing.template dim<0>()-1,I1,J1)));

            }

            // 10 (H,E)+(E,H)
            I1=0;
            J1=indexing.template dim<1>()-1;
            I2=indexing.template dim<0>()-1;
            for(short_t J2=1; J2<indexing.template dim<1>()-1; J2++)
            {

                eval(out(dof1+indexing.index(I1,J1,0),dof2+indexing.index(I2,J2,0))) +=
                        eval(in2(i-1,dof1+indexing.index(I1,J1,indexing.template dim<2>()-1),dof2+indexing.index(I2,J2,indexing.template dim<2>()-1)));
                eval(out(dof1+indexing.index(I2,J2,0),dof2+indexing.index(I1,J1,0))) +=
                        eval(in2(i-1,dof1+indexing.index(I2,J2,indexing.template dim<2>()-1),dof2+indexing.index(I1,J1,indexing.template dim<2>()-1)));

                eval(out(dof1+indexing.index(J1,0,I1),dof2+indexing.index(J2,0,I2))) +=
                        eval(in2(j-1,dof1+indexing.index(J1,indexing.template dim<1>()-1,I1),dof2+indexing.index(J2,indexing.template dim<1>()-1,I2)));
                eval(out(dof1+indexing.index(J2,0,I2),dof2+indexing.index(J1,0,I1))) +=
                        eval(in2(j-1,dof1+indexing.index(J2,indexing.template dim<1>()-1,I2),dof2+indexing.index(J1,indexing.template dim<1>()-1,I1)));

                eval(out(dof1+indexing.index(0,I1,J1),dof2+indexing.index(0,I2,J2))) +=
                        eval(in2(k-1,dof1+indexing.index(indexing.template dim<0>()-1,I1,J1),dof2+indexing.index(indexing.template dim<0>()-1,I2,J2)));
                eval(out(dof1+indexing.index(0,I2,J2),dof2+indexing.index(0,I1,J1))) +=
                        eval(in2(k-1,dof1+indexing.index(indexing.template dim<0>()-1,I2,J2),dof2+indexing.index(indexing.template dim<0>()-1,I1,J1)));

            }

            // 11 (F,B+G)+(B+G,F)
            I1=0;
            J1=0;
            J2=0;
            for(short_t I2=1; I2<indexing.template dim<0>(); I2++)
            {

                eval(out(dof1+indexing.index(I1,J1,0),dof2+indexing.index(I2,J2,0))) +=
                        eval(in2(i-1,dof1+indexing.index(I1,J1,indexing.template dim<2>()-1),dof2+indexing.index(I2,J2,indexing.template dim<2>()-1))) +
                        eval(in2(j-1,dof1+indexing.index(I1,indexing.template dim<1>()-1,0),dof2+indexing.index(I2,indexing.template dim<1>()-1,0))) +
                        eval(in2(i-1,j-1,dof1+indexing.index(I1,indexing.template dim<1>()-1,indexing.template dim<2>()-1),dof2+indexing.index(I2,indexing.template dim<1>()-1,indexing.template dim<2>()-1)));

                eval(out(dof1+indexing.index(I2,J2,0),dof2+indexing.index(I1,J1,0))) +=
                        eval(in2(i-1,dof1+indexing.index(I2,J2,indexing.template dim<2>()-1),dof2+indexing.index(I1,J1,indexing.template dim<2>()-1))) +
                        eval(in2(j-1,dof1+indexing.index(I2,indexing.template dim<1>()-1,0),dof2+indexing.index(I1,indexing.template dim<1>()-1,0))) +
                        eval(in2(i-1,j-1,dof1+indexing.index(I2,indexing.template dim<1>()-1,indexing.template dim<2>()-1),dof2+indexing.index(I1,indexing.template dim<1>()-1,indexing.template dim<2>()-1)));

                eval(out(dof1+indexing.index(J1,0,I1),dof2+indexing.index(J2,0,I2))) +=
                        eval(in2(j-1,dof1+indexing.index(J1,indexing.template dim<1>()-1,I1),dof2+indexing.index(J2,indexing.template dim<1>()-1,I2))) +
                        eval(in2(k-1,dof1+indexing.index(indexing.template dim<0>()-1,0,I1),dof2+indexing.index(indexing.template dim<0>()-1,0,I2))) +
                        eval(in2(j-1,k-1,dof1+indexing.index(indexing.template dim<0>()-1,indexing.template dim<1>()-1,I1),dof2+indexing.index(indexing.template dim<0>()-1,indexing.template dim<1>()-1,I2)));

                eval(out(dof1+indexing.index(J2,0,I2),dof2+indexing.index(J1,0,I1))) +=
                        eval(in2(j-1,dof1+indexing.index(J2,indexing.template dim<1>()-1,I2),dof2+indexing.index(J1,indexing.template dim<1>()-1,I1))) +
                        eval(in2(k-1,dof1+indexing.index(indexing.template dim<0>()-1,0,I2),dof2+indexing.index(indexing.template dim<0>()-1,0,I1))) +
                        eval(in2(j-1,k-1,dof1+indexing.index(indexing.template dim<0>()-1,indexing.template dim<1>()-1,I2),dof2+indexing.index(indexing.template dim<0>()-1,indexing.template dim<1>()-1,I1)));

                eval(out(dof1+indexing.index(0,I1,J1),dof2+indexing.index(0,I2,J2))) +=
                        eval(in2(k-1,dof1+indexing.index(indexing.template dim<0>()-1,I1,J1),dof2+indexing.index(indexing.template dim<0>()-1,I2,J2))) +
                        eval(in2(i-1,dof1+indexing.index(0,I1,indexing.template dim<2>()-1),dof2+indexing.index(0,I2,indexing.template dim<2>()-1))) +
                        eval(in2(i-1,k-1,dof1+indexing.index(indexing.template dim<0>()-1,I1,indexing.template dim<2>()-1),dof2+indexing.index(indexing.template dim<0>()-1,I2,indexing.template dim<2>()-1)));

                eval(out(dof1+indexing.index(0,I2,J2),dof2+indexing.index(0,I1,J1))) +=
                        eval(in2(k-1,dof1+indexing.index(indexing.template dim<0>()-1,I2,J2),dof2+indexing.index(indexing.template dim<0>()-1,I1,J1))) +
                        eval(in2(i-1,dof1+indexing.index(0,I2,indexing.template dim<2>()-1),dof2+indexing.index(0,I1,indexing.template dim<2>()-1))) +
                        eval(in2(i-1,k-1,dof1+indexing.index(indexing.template dim<0>()-1,I2,indexing.template dim<2>()-1),dof2+indexing.index(indexing.template dim<0>()-1,I1,indexing.template dim<2>()-1)));


            }

            // 12 (G,B)+(B,G)
            I1=indexing.template dim<0>()-1;
            J1=0;
            J2=0;
            for(short_t I2=1; I2<indexing.template dim<0>()-1; I2++)
            {

                eval(out(dof1+indexing.index(I1,J1,0),dof2+indexing.index(I2,J2,0))) +=
                        eval(in2(i-1,dof1+indexing.index(I1,J1,indexing.template dim<2>()-1),dof2+indexing.index(I2,J2,indexing.template dim<2>()-1))) +
                        eval(in2(j-1,dof1+indexing.index(I1,indexing.template dim<1>()-1,0),dof2+indexing.index(I2,indexing.template dim<1>()-1,0))) +
                        eval(in2(i-1,j-1,dof1+indexing.index(I1,indexing.template dim<1>()-1,indexing.template dim<2>()-1),dof2+indexing.index(I2,indexing.template dim<1>()-1,indexing.template dim<2>()-1)));

                eval(out(dof1+indexing.index(I2,J2,0),dof2+indexing.index(I1,J1,0))) +=
                        eval(in2(i-1,dof1+indexing.index(I2,J2,indexing.template dim<2>()-1),dof2+indexing.index(I1,J1,indexing.template dim<2>()-1))) +
                        eval(in2(j-1,dof1+indexing.index(I2,indexing.template dim<1>()-1,0),dof2+indexing.index(I1,indexing.template dim<1>()-1,0))) +
                        eval(in2(i-1,j-1,dof1+indexing.index(I2,indexing.template dim<1>()-1,indexing.template dim<2>()-1),dof2+indexing.index(I1,indexing.template dim<1>()-1,indexing.template dim<2>()-1)));

                eval(out(dof1+indexing.index(J1,0,I1),dof2+indexing.index(J2,0,I2))) +=
                        eval(in2(j-1,dof1+indexing.index(J1,indexing.template dim<1>()-1,I1),dof2+indexing.index(J2,indexing.template dim<1>()-1,I2))) +
                        eval(in2(k-1,dof1+indexing.index(indexing.template dim<0>()-1,0,I1),dof2+indexing.index(indexing.template dim<0>()-1,0,I2))) +
                        eval(in2(j-1,k-1,dof1+indexing.index(indexing.template dim<0>()-1,indexing.template dim<1>()-1,I1),dof2+indexing.index(indexing.template dim<0>()-1,indexing.template dim<1>()-1,I2)));

                eval(out(dof1+indexing.index(J2,0,I2),dof2+indexing.index(J1,0,I1))) +=
                        eval(in2(j-1,dof1+indexing.index(J2,indexing.template dim<1>()-1,I2),dof2+indexing.index(J1,indexing.template dim<1>()-1,I1))) +
                        eval(in2(k-1,dof1+indexing.index(indexing.template dim<0>()-1,0,I2),dof2+indexing.index(indexing.template dim<0>()-1,0,I1))) +
                        eval(in2(j-1,k-1,dof1+indexing.index(indexing.template dim<0>()-1,indexing.template dim<1>()-1,I2),dof2+indexing.index(indexing.template dim<0>()-1,indexing.template dim<1>()-1,I1)));

                eval(out(dof1+indexing.index(0,I1,J1),dof2+indexing.index(0,I2,J2))) +=
                        eval(in2(k-1,dof1+indexing.index(indexing.template dim<0>()-1,I1,J1),dof2+indexing.index(indexing.template dim<0>()-1,I2,J2))) +
                        eval(in2(i-1,dof1+indexing.index(0,I1,indexing.template dim<2>()-1),dof2+indexing.index(0,I2,indexing.template dim<2>()-1))) +
                        eval(in2(i-1,k-1,dof1+indexing.index(indexing.template dim<0>()-1,I1,indexing.template dim<2>()-1),dof2+indexing.index(indexing.template dim<0>()-1,I2,indexing.template dim<2>()-1)));

                eval(out(dof1+indexing.index(0,I2,J2),dof2+indexing.index(0,I1,J1))) +=
                        eval(in2(k-1,dof1+indexing.index(indexing.template dim<0>()-1,I2,J2),dof2+indexing.index(indexing.template dim<0>()-1,I1,J1))) +
                        eval(in2(i-1,dof1+indexing.index(0,I2,indexing.template dim<2>()-1),dof2+indexing.index(0,I1,indexing.template dim<2>()-1))) +
                        eval(in2(i-1,k-1,dof1+indexing.index(indexing.template dim<0>()-1,I2,indexing.template dim<2>()-1),dof2+indexing.index(indexing.template dim<0>()-1,I1,indexing.template dim<2>()-1)));

            }

            // 13 (B,B)
            J1=0;
            J2=0;
            for(short_t I1=1; I1<indexing.template dim<0>()-1; I1++)
                for(short_t I2=1; I2<indexing.template dim<0>()-1; I2++)
                {

                    eval(out(dof1+indexing.index(I1,J1,0),dof2+indexing.index(I2,J2,0))) +=
                            eval(in2(i-1,dof1+indexing.index(I1,J1,indexing.template dim<2>()-1),dof2+indexing.index(I2,J2,indexing.template dim<2>()-1))) +
                            eval(in2(j-1,dof1+indexing.index(I1,indexing.template dim<1>()-1,0),dof2+indexing.index(I2,indexing.template dim<1>()-1,0))) +
                            eval(in2(i-1,j-1,dof1+indexing.index(I1,indexing.template dim<1>()-1,indexing.template dim<2>()-1),dof2+indexing.index(I2,indexing.template dim<1>()-1,indexing.template dim<2>()-1)));

                    eval(out(dof1+indexing.index(J1,0,I1),dof2+indexing.index(J2,0,I2))) +=
                            eval(in2(j-1,dof1+indexing.index(J1,indexing.template dim<1>()-1,I1),dof2+indexing.index(J2,indexing.template dim<1>()-1,I2))) +
                            eval(in2(k-1,dof1+indexing.index(indexing.template dim<0>()-1,0,I1),dof2+indexing.index(indexing.template dim<0>()-1,0,I2))) +
                            eval(in2(j-1,k-1,dof1+indexing.index(indexing.template dim<0>()-1,indexing.template dim<1>()-1,I1),dof2+indexing.index(indexing.template dim<0>()-1,indexing.template dim<1>()-1,I2)));

                    eval(out(dof1+indexing.index(0,I1,J1),dof2+indexing.index(0,I2,J2))) +=
                            eval(in2(k-1,dof1+indexing.index(indexing.template dim<0>()-1,I1,J1),dof2+indexing.index(indexing.template dim<0>()-1,I2,J2))) +
                            eval(in2(i-1,dof1+indexing.index(0,I1,indexing.template dim<2>()-1),dof2+indexing.index(0,I2,indexing.template dim<2>()-1))) +
                            eval(in2(i-1,k-1,dof1+indexing.index(indexing.template dim<0>()-1,I1,indexing.template dim<2>()-1),dof2+indexing.index(indexing.template dim<0>()-1,I2,indexing.template dim<2>()-1)));

                }

            // 14 (F,F)
            eval(out(dof1+0,dof2+0)) +=
                    eval(in2(i-1,dof1+indexing.index(0,0,indexing.template dim<2>()-1),dof2+indexing.index(0,0,indexing.template dim<2>()-1))) +
                    eval(in2(j-1,dof1+indexing.index(0,indexing.template dim<1>()-1,0),dof2+indexing.index(0,indexing.template dim<1>()-1,0))) +
                    eval(in2(i-1,j-1,dof1+indexing.index(0,indexing.template dim<1>()-1,indexing.template dim<2>()-1),dof2+indexing.index(0,indexing.template dim<1>()-1,indexing.template dim<2>()-1))) +
                    eval(in2(k-1,dof1+indexing.index(indexing.template dim<0>()-1,0,0),dof2+indexing.index(indexing.template dim<0>()-1,0,0))) +
                    eval(in2(i-1,k-1,dof1+indexing.index(indexing.template dim<0>()-1,0,indexing.template dim<2>()-1),dof2+indexing.index(indexing.template dim<0>()-1,0,indexing.template dim<2>()-1))) +
                    eval(in2(j-1,k-1,dof1+indexing.index(indexing.template dim<0>()-1,indexing.template dim<1>()-1,0),dof2+indexing.index(indexing.template dim<0>()-1,indexing.template dim<1>()-1,0))) +
                    eval(in2(i-1,j-1,k-1,
                             dof1+indexing.index(indexing.template dim<0>()-1,indexing.template dim<1>()-1,indexing.template dim<2>()-1),
                             dof2+indexing.index(indexing.template dim<0>()-1,indexing.template dim<1>()-1,indexing.template dim<2>()-1)));

        }
    };
    } // namespace functors
} // namespace gdl
