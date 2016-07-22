#pragma once

namespace gdl{
    namespace functors{

        /** takes the boundary trace of the vector field beta, and projects it along the direction
         specified by the input vector field int_normals.*/
        struct project_on_boundary {

            using beta=gt::accessor<0, enumtype::in, gt::extent<> , 5>;
            using int_normals=gt::accessor<1, enumtype::in, gt::extent<> , 6>;
            using bd_mass=gt::accessor<2, enumtype::in, gt::extent<> , 6>;
            using out=gt::accessor<3, enumtype::inout, gt::extent<> , 5>;
            using arg_list=boost::mpl::vector<beta, int_normals, bd_mass, out> ;

            template <typename Evaluation>
            GT_FUNCTION
            static void Do(Evaluation const & eval, x_interval) {
                gt::dimension<4>::Index row;
                gt::dimension<5>::Index face;
                gt::dimension<5>::Index col;
                gt::dimension<5>::Index dim;
                gt::dimension<6>::Index Mface;

                const auto N_dofs=eval.template get_storage_dim<3>(out());//N_DOFS
                const auto N_faces=eval.template get_storage_dim<5>(int_normals());//6
                const auto N_dims=eval.template get_storage_dim<4>(int_normals());//3

                for (uint_t dof = 0; dof<N_dofs; ++dof)
                    for (uint_t k = 0; k<N_dofs; ++k)
                        for (uint_t f =0; f<N_faces; ++f)
                        {
                            float_type product = 0.;
                            for (uint_t d =0; d<N_dims; ++d)
                                product += eval(beta(row+k, dim+d)
                                               * int_normals(row+k, dim+d, Mface+f));
                            eval(out(row+dof, face+f)) += eval(bd_mass(row+dof, col+k, Mface+f))*product ;
                        }


            }
        };
    } //namespace functors
} //namespace gdl
