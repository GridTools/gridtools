#pragma once

namespace gdl {

    namespace functors{

        typedef gridtools::interval<gridtools::level<0,-1>, gridtools::level<1,-1> > x_interval;

        // [partial_reduction]
        /**
           @class functor performing partial reduction of a vector

           Given a mesh and, a dof distribution on the mesh element and a vector assigned to each element whose element
           correspond to the single dofs, this functor performs an element-wise reduction of the vector itself applying
           the provided operator. As for the assemble procedure, each mesh element is responsible only for a given subset
           of the available dofs, the same policy is applied. The reduction result is stored in the first vector element.
           The final reduction value must be calculated outside looping over the first elements of the vector.
         */
        // TODO: other element-shape specific reduction are needed
        // TODO: the implemented partial reduction is based on dof/element identification described in assembly functors
        // TODO: in place and out of place can be handled providing an operator for selecting between = and +=
        // TODO: part of the code handling the inclusion/exclusion of elements for reduction calculation is
        // duplicated in assembly code, can we factorize it?
        template <ushort_t N_DOF0, ushort_t N_DOF1, ushort_t N_DOF2, typename Operator>
        struct partial_hexahedron_assembled_reduction {

            using in=gt::accessor<0, enumtype::in, gt::extent<> , 4> ;
            using out=gt::accessor<1, enumtype::inout, gt::extent<> , 4> ;
            using arg_list=boost::mpl::vector<in, out> ;

            template <typename Evaluation>
            GT_FUNCTION
            static void Do(Evaluation const & eval, x_interval) {
                gt::dimension<4>::Index row;
                constexpr gt::meta_storage_base<static_int<__COUNTER__>,gt::layout_map<2,1,0>,false> indexing{N_DOF0,N_DOF1,N_DOF2};
                uint_t const num_rows=eval.get().template get_storage_dim<3>(in());

                // Loop over vector elements

                // 4 F
                auto red_val = eval(in(row+0));

                // 0 INTERNAL NON SHARED DOFS
                for(short_t I1=1; I1<indexing.template dim<0>()-1; I1++)
                    for(short_t J1=1; J1<indexing.template dim<1>()-1; J1++)
                        for(short_t K1=1; K1<indexing.template dim<2>()-1; K1++)
                            red_val = Operator::eval(red_val,eval(in(row+indexing.index(I1,J1,K1))));

                // TODO: in this case these loops have less sense wrt assemble case, merge them
                // 1 A
                for(short_t I1=1; I1<indexing.template dim<0>()-1; I1++)
                    for(short_t J1=1; J1<indexing.template dim<1>()-1; J1++)
                    {
                        red_val = Operator::eval(red_val,eval(in(row+indexing.index(I1,J1,0))));

                        red_val = Operator::eval(red_val,eval(in(row+indexing.index(J1,0,I1))));

                        red_val = Operator::eval(red_val,eval(in(row+indexing.index(0,I1,J1))));
                    }

                // 2 B
                short_t J1=0;
                for(short_t I1=1; I1<indexing.template dim<0>()-1; I1++)
                {
                    red_val = Operator::eval(red_val,eval(in(row+indexing.index(I1,J1,0))));

                    red_val = Operator::eval(red_val,eval(in(row+indexing.index(J1,0,I1))));

                    red_val = Operator::eval(red_val,eval(in(row+indexing.index(0,I1,J1))));
                }

                eval(out(row+0)) = red_val;

            }
        };
    }
}
