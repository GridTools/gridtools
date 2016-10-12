#pragma once

namespace gdl {

    namespace functors {

        typedef gridtools::interval<gridtools::level<0,-1>, gridtools::level<1,-1> > x_interval;

        // TODO: add halo as template par, if it cannot be asked to the containers!
        struct richardson_iteration {

            using matrix=gt::accessor<0, enumtype::in, gt::extent<> , 3>;
            using rhs=gt::accessor<1, enumtype::in, gt::extent<> , 3> ;
            using scaling=gt::accessor<2, enumtype::in, gt::extent<> , 3> ;// TODO: how can be passed in input a scalar (floating point) parameter?
            using unknowns=gt::accessor<3, enumtype::in, gt::extent<> , 3> ;
            using unknowns_new=gt::accessor<4, enumtype::inout, gt::extent<> , 3> ;
            using arg_list=boost::mpl::vector<matrix, rhs, scaling, unknowns, unknowns_new> ;


            template <typename Evaluation>
            GT_FUNCTION
            static void Do(Evaluation const & eval, x_interval) {

                const uint_t size = eval.get().template get_storage_dims<0>(rhs());
                const uint_t n_problems = eval.get().template get_storage_dims<1>(rhs());

//                GRIDTOOLS_STATIC_ASSERT(size==eval.get().template get_storage_dims<1>(matrix()), "GDL Error: ???");
//                GRIDTOOLS_STATIC_ASSERT(eval.get().template get_storage_dims<0>(matrix())==eval.get().template get_storage_dims<0>(unkowns()), "GDL Error: ???");
//                GRIDTOOLS_STATIC_ASSERT(n_problems==eval.get().template get_storage_dims<1>(unkowns()), "GDL Error: ???");

                gt::dimension<1> row_vec;
                gt::dimension<2> col_matrix;
                gt::dimension<2> probl_vec;
                gt::dimension<3> last_vec;


                // TODO: do not forgot halo!!!
                // TODO: this loop is similar to the one for assembly, we should have basic vectorial algebra functor somewhere (wAx+zB)
                eval(unknowns_new()) = 0.;
                for(uint_t i=0;i<size;++i) {
                    eval(unknowns_new()) += eval(matrix(col_matrix+i)*!unknowns(row_vec+i,probl_vec+0,last_vec+0));
                }

//                eval(unknowns_new()) = eval(!scaling(0,0,0)*(rhs() - unknowns_new()));
                eval(unknowns_new()) = eval(!scaling(0,0,0)*(rhs() - unknowns_new()) + unknowns());

            }

        };

    }

}
