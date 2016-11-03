#include <common/gt_math.hpp>
#include <Intrepid_HGRAD_LINE_Cn_FEM_JACOBI.hpp>
#include <common/gt_math.hpp>
#include <cmath>
// [includes]
#include <Intrepid_FunctionSpaceTools.hpp>
#include <Intrepid_Types.hpp>

#include "cubature.hpp"
// [includes]

namespace gdl{

    template <uint_t Dim, uint_t Order>
    struct evaluate_polynomial;

    template <int_t Order>
    struct evaluate_polynomial< 3, Order >{

        template <typename Storage>
        static void apply(Storage& storage_, Intrepid::FieldContainer<gt::float_type> const& val_per_line_, int_t on_boundary_){

            switch (on_boundary_){
            case 0 :

            {
                uint_t m=0;
                for(int_t i=0; i<(Order); ++i)
                    for(int_t j=0; j<(Order)-i; ++j)
                        for(int_t k=0; k<(Order)-i-j; ++k)
                        {
                            uint_t q=0;
                            //hypothesis: tensor product quadrature grid
                            //hypothesis: on the layout of the quadrature points
                            for(int_t qz=0; qz<val_per_line_.dimension(1); ++qz)
                                for(int_t qy=0; qy<val_per_line_.dimension(1); ++qy)
                                    for(int_t qx=0; qx<val_per_line_.dimension(1); ++qx)
                                    {
                                        // printf("storage(%d, %d) = %f*%f*%f\n",m,q, val_per_line_(i, qx), val_per_line_(j, qx), val_per_line_(k, qx));
                                        storage_(m,q) =
                                        val_per_line_(i, qx)  *
                                            val_per_line_(j, qy) *
                                            val_per_line_(k, qz)
                                            ;
                                        ++q;
                                    }
                            ++m;
                        }
                break;
            }

            case 1 :

            {
                uint_t m=0;
                for(int_t i=0; i<(Order); ++i)
                    for(int_t j=0; j<(Order)-i; ++j)
                        for(int_t k=0; k<(Order)-i-j; ++k)
                        {
                            uint_t q=0;
                            //hypothesis: tensor product quadrature grid
                            //hypothesis: on the layout of the quadrature points
                            for(int_t qz=0; qz<val_per_line_.dimension(1); ++qz)
                                for(int_t qy=0; qy<val_per_line_.dimension(1); ++qy)
                                    {
                                        // printf("storage(%d, %d) = %f*%f*%f\n",m,q, val_per_line_(i, qx), val_per_line_(j, qx), val_per_line_(k, qx));
                                        storage_(m,q) =
                                        val_per_line_(i, 0)  *
                                            val_per_line_(j, qy) *
                                            val_per_line_(k, qz)
                                            ;
                                        ++q;
                                    }
                            ++m;
                        }
                break;
            }

            case 2 :

            {
                uint_t m=0;
                for(int_t i=0; i<(Order); ++i)
                    for(int_t j=0; j<(Order)-i; ++j)
                        for(int_t k=0; k<(Order)-i-j; ++k)
                        {
                            uint_t q=0;
                            //hypothesis: tensor product quadrature grid
                            //hypothesis: on the layout of the quadrature points
                            for(int_t qz=0; qz<val_per_line_.dimension(1); ++qz)
                                for(int_t qy=0; qy<val_per_line_.dimension(1); ++qy)
                                    {
                                        // printf("storage(%d, %d) = %f*%f*%f\n",m,q, val_per_line_(i, qx), val_per_line_(j, qx), val_per_line_(k, qx));
                                        storage_(m,q) =
                                        val_per_line_(i, val_per_line_.dimension(1)-1)  *
                                            val_per_line_(j, qy) *
                                            val_per_line_(k, qz)
                                            ;
                                        ++q;
                                    }
                            ++m;
                        }
                break;
            }

            case 4 :

            {
                uint_t m=0;
                for(int_t i=0; i<(Order); ++i)
                    for(int_t j=0; j<(Order)-i; ++j)
                        for(int_t k=0; k<(Order)-i-j; ++k)
                        {
                            uint_t q=0;
                            //hypothesis: tensor product quadrature grid
                            //hypothesis: on the layout of the quadrature points
                            for(int_t qz=0; qz<val_per_line_.dimension(1); ++qz)
                                for(int_t qx=0; qx<val_per_line_.dimension(1); ++qx)
                                    {
                                        // printf("storage(%d, %d) = %f*%f*%f\n",m,q, val_per_line_(i, qx), val_per_line_(j, qx), val_per_line_(k, qx));
                                        storage_(m,q) =
                                        val_per_line_(i, qx)  *
                                            val_per_line_(j, 0) *
                                            val_per_line_(k, qz)
                                            ;
                                        ++q;
                                    }
                            ++m;
                        }
                break;
            }

            case 8 :

            {
                uint_t m=0;
                for(int_t i=0; i<(Order); ++i)
                    for(int_t j=0; j<(Order)-i; ++j)
                        for(int_t k=0; k<(Order)-i-j; ++k)
                        {
                            uint_t q=0;
                            //hypothesis: tensor product quadrature grid
                            //hypothesis: on the layout of the quadrature points
                            for(int_t qz=0; qz<val_per_line_.dimension(1); ++qz)
                                for(int_t qx=0; qx<val_per_line_.dimension(1); ++qx)
                                    {
                                        // printf("storage(%d, %d) = %f*%f*%f\n",m,q, val_per_line_(i, qx), val_per_line_(j, qx), val_per_line_(k, qx));
                                        storage_(m,q) =
                                        val_per_line_(i, qx)  *
                                            val_per_line_(j, val_per_line_.dimension(1)-1) *
                                            val_per_line_(k, qz)
                                            ;
                                        ++q;
                                    }
                            ++m;
                        }
                break;
            }

            case 16 :

            {
                uint_t m=0;
                for(int_t i=0; i<(Order); ++i)
                    for(int_t j=0; j<(Order)-i; ++j)
                        for(int_t k=0; k<(Order)-i-j; ++k)
                        {
                            uint_t q=0;
                            //hypothesis: tensor product quadrature grid
                            //hypothesis: on the layout of the quadrature points
                            for(int_t qy=0; qy<val_per_line_.dimension(1); ++qy)
                                for(int_t qx=0; qx<val_per_line_.dimension(1); ++qx)
                                    {
                                        // printf("storage(%d, %d) = %f*%f*%f\n",m,q, val_per_line_(i, qx), val_per_line_(j, qx), val_per_line_(k, qx));
                                        storage_(m,q) =
                                        val_per_line_(i, qx)  *
                                            val_per_line_(j, qy) *
                                            val_per_line_(k, 0)
                                            ;
                                        ++q;
                                    }
                            ++m;
                        }
                break;
            }

            case 32 :

            {
                uint_t m=0;
                for(int_t i=0; i<(Order); ++i)
                    for(int_t j=0; j<(Order)-i; ++j)
                        for(int_t k=0; k<(Order)-i-j; ++k)
                        {
                            uint_t q=0;
                            //hypothesis: tensor product quadrature grid
                            //hypothesis: on the layout of the quadrature points
                            for(int_t qy=0; qy<val_per_line_.dimension(1); ++qy)
                                for(int_t qx=0; qx<val_per_line_.dimension(1); ++qx)
                                    {
                                        // printf("storage(%d, %d) = %f*%f*%f\n",m,q, val_per_line_(i, qx), val_per_line_(j, qx), val_per_line_(k, qx));
                                        storage_(m,q) =
                                        val_per_line_(i, qx)  *
                                            val_per_line_(j, qy) *
                                            val_per_line_(k, val_per_line_.dimension(1)-1)
                                            ;
                                        ++q;
                                    }
                            ++m;
                        }
                break;
            }

            default :
                assert(false);
            }
        }

        template <typename Storage>
        static void derivative(Storage& storage_, Intrepid::FieldContainer<gt::float_type> const& val_per_line_, Intrepid::FieldContainer<gt::float_type> const& der_per_line_, int on_boundary_){


            uint_t m=0;
            for(int_t i=0; i<(Order); ++i)
                for(int_t j=0; j<(Order)-i; ++j)
                    for(int_t k=0; k<(Order)-i-j; ++k)
                    {
                        uint_t q=0;
                        //hypothesis: tensor product quadrature grid
                        //hypothesis: on the layout of the quadrature points
                        for(int_t qz=0; qz<val_per_line_.dimension(1); ++qz)
                            for(int_t qy=0; qy<val_per_line_.dimension(1); ++qy)
                                for(int_t qx=0; qx<val_per_line_.dimension(1); ++qx)
                                {
                                    // printf("storage(%d, %d) = %f*%f*%f\n",m,q, val_per_line_(i, qx), val_per_line_(j, qx), val_per_line_(k, qx));
                                    storage_(m,q,0) =
                                        der_per_line_(i, qx, 0)  *
                                        val_per_line_(j, qy)  *
                                        val_per_line_(k, qz)
                                        ;
                                    storage_(m,q,1) =
                                        val_per_line_(i, qx)  *
                                        der_per_line_(j, qy, 0)  *
                                        val_per_line_(k, qz)
                                        ;
                                    storage_(m,q,2) =
                                        val_per_line_(i, qx)  *
                                        val_per_line_(j, qy)  *
                                        der_per_line_(k, qz, 0)
                                        ;
                                    ++q;
                                }
                        ++m;
                    }

        }
    };


    template <uint_t Order>
    struct evaluate_polynomial<2, Order >{
        template <typename Storage>
        static void apply(Storage& storage_, Intrepid::FieldContainer<gt::float_type> const& val_per_line_){


            uint_t m=0;
            for(int_t i=0; i<(Order); ++i)
                for(int_t j=0; j<(Order)-i; ++j)
                    {
                        uint_t q=0;
                        //hypothesis: tensor product quadrature grid
                        //hypothesis: on the layout of the quadrature points
                        for(int_t qy=0; qy<val_per_line_.dimension(1); ++qy)
                            for(int_t qx=0; qx<val_per_line_.dimension(1); ++qx)
                            {
                                // printf("storage(%d, %d) = %f*%f*%f\n",m,q, val_per_line_(i, qx), val_per_line_(j, qx), val_per_line_(k, qx));
                                storage_(m,q) =
                                    val_per_line_(i, qx)  *
                                    val_per_line_(j, qy)
                                    ;
                                ++q;
                            }
                        ++m;
                    }

        }

        template <typename Storage>
        static void derivative(Storage& storage_, Intrepid::FieldContainer<gt::float_type> val_per_line_, Intrepid::FieldContainer<gt::float_type> der_per_line_){


            uint_t m=0;
            for(int_t i=0; i<(Order); ++i)
                for(int_t j=0; j<(Order)-i; ++j)
                    {
                        uint_t q=0;
                        //hypothesis: tensor product quadrature grid
                        //hypothesis: on the layout of the quadrature points
                        for(int_t qy=0; qy<val_per_line_.dimension(1); ++qy)
                            for(int_t qx=0; qx<val_per_line_.dimension(1); ++qx)
                            {
                                // printf("storage(%d, %d) = %f*%f*%f\n",m,q, val_per_line_(i, qx), val_per_line_(j, qx), val_per_line_(k, qx));
                                storage_(m,q,0) =
                                    der_per_line_(i, qx,0)  *
                                    val_per_line_(j, qy)
                                    ;
                                storage_(m,q,1) =
                                    val_per_line_(i, qx)  *
                                    der_per_line_(j, qy,0)
                                    ;
                                ++q;
                            }
                        ++m;
                    }

            // uint_t m=0;

            // for(int_t i=0; i<Order; ++i)
            //     for(int_t j=0; j<Order-i; ++j)
            //     {
            //         for(int_t q=0; q<quad_points_.dimension(0); ++q){
            //             auto qx = quad_points_(q, 0);
            //             auto qy = quad_points_(q, 1);
            //             storage_(m,q,0)=(val_per_line_(i-1, qx) + val_per_line_(i, qx))/(1-qx*qx) *
            //                 val_per_line_(j, qy);

            //             storage_(m,q,1)=(val_per_line_(j-1, qy) + val_per_line_(j, qy))/(1-qy*qy) *
            //                 val_per_line_(i, qx);

            //         }
            //         ++m;
            //     }
        }
    };

    template <uint_t Order>
    struct evaluate_polynomial< 1, Order >{
        template <typename Storage, typename Quad>
        static void apply(Storage& storage_, Quad const& quad_points_, Intrepid::FieldContainer<gt::float_type> const& val_per_line_){
            assert(false); //1D not supported
        }


        template <typename Storage, typename Quad>
        static void derivative(Storage& storage_, Quad const& quad_points_, Intrepid::FieldContainer<gt::float_type> const& val_per_line_){
            assert(false); //1D not supported
        }
    };

    template <uint_t Dim, uint_t Order>
    struct legendre{

    private:
        Intrepid::Basis_HGRAD_LINE_Cn_FEM_JACOBI<gt::float_type, Intrepid::FieldContainer<gt::float_type> > line_polynomial;
        // std::unique_ptr<Intrepid::FieldArray> line_values;
    public:

        constexpr legendre():
            line_polynomial(Order,0,0)
            // , line_values()
        {} // Jacobi poynomial with alpha=beta=0

        template <typename Storage>
        void getDofCoords(Storage& /*s*/) const {
            // should fill the input storage with the knots local coordinates
            // to be implemented
            assert(false);
        }

        /** compile-time known*/
        constexpr int getCardinality() const
        {
            //returns the number of basis functions (P)^dim
            return factorial<Order+Dim>::value/(factorial<Order>::value * factorial<Dim>::value);
        }

        /**
           @brief compute the values of an operator on the basis functions, evaluate
           on quadrature points

           @tparam Storage arbitrary storage type for the output values
           @tparam Quad arbitrary storage type for the quadrature points
           (might differ from the previous one)
         */
        template <typename Storage, typename Quad>
        void getValues(Storage& storage_, Quad const& quad_points_, Intrepid::EOperator op) const
        {

            // typedef cubature<QOrder+1, enumtype::Line> cub;
            // Intrepid::FieldContainer<gt::float_type> cub_points_i(cub::numCubPoints(), 1);
            // Intrepid::FieldContainer<gt::float_type> cub_weights_i(cub::numCubPoints());
            // cub::cub()->getCubature(cub_points_i, cub_weights_i);

            // Important hypothesis: the quadrature points form a tensor product grid on the element
            // valid only for hypercube meshes

            //make sure that the cub points are well ordered
            std::vector<uint_t> permutations(quad_points_.dimension(0));
            for(uint_t i=0; i< quad_points_.dimension(0); ++i){
                permutations[i]=i;
            }

            std::vector<gt::float_type> to_reorder( quad_points_.dimension(0) );

            // fill in the reorder vector such that the larger numbers correspond to larger strides
            for(uint_t i=0; i<quad_points_.dimension(0); ++i){
                to_reorder[i]=( (quad_points_(i, 0)+2.) +
                                (quad_points_(i, 1)+2.)*4 +
                                (quad_points_(i, 2)+2.)*16);
            }

            std::sort(permutations.begin(), permutations.end(),
                      [&to_reorder](uint_t a, uint_t b){
                          return to_reorder[a]>to_reorder[b];
                      } );

            Quad ordered_quad_points_(quad_points_);
            for(uint_t k=0; k<quad_points_.dimension(0); ++k){
                ordered_quad_points_(k,0)=quad_points_(permutations[k],0);
                ordered_quad_points_(k,1)=quad_points_(permutations[k],1);
                ordered_quad_points_(k,2)=quad_points_(permutations[k],2);
            }

            //method to find the number of quadrature points on a line
            //(when the y and z coordinates stay the same we increment by 1)
            //(or we suppose that the quad points are well ordered)
            // uint_t cubic_root = 0;
            // for(uint_t k=1; k<quad_points_.dimension(0); ++k){
            //     if(ordered_quad_points_(k,1) != ordered_quad_points_(k-1,1))
            //         {
            //             cubic_root = k;
            //             break;
            //         }
            //     }
            uint_t cubic_root = 0;

            auto val_x = ordered_quad_points_(0,0);
            auto val_y = ordered_quad_points_(0,1);
            auto val_z = ordered_quad_points_(0,2);

            int_t on_boundary = (val_x==1.) + (val_x==-1.)*(2) + (val_y==1.)*4 + (val_y==-1.)*(8) + (val_z==1.)*16 + (val_z==-1.)*(32);

            if(val_x != 1. && val_x != -1.)
            for(uint_t k=1; k<quad_points_.dimension(0); ++k){
                if(
                    ordered_quad_points_(k,0) == val_x
                    )
                    {
                        cubic_root = k;
                        break;
                    }
            }
            else  // this happens e.g. in case all the quad points lay on the yz surface (for the boundary integrals)
                for(uint_t k=1; k<quad_points_.dimension(0); ++k){
                    if(
                        ordered_quad_points_(k,1) != val_y
                        )
                    {
                        cubic_root = k;
                        break;
                    }
                }

            assert(cubic_root); // this happens e.g. in case all the quad points lay on an edge

            Intrepid::FieldContainer<gt::float_type> cub_points_i(cubic_root, 1);
            for(uint_t k=0; k<cubic_root; ++k){
                cub_points_i(k,0)=ordered_quad_points_(k,0);
            }

// #ifndef NDEBUG
//             uint_t m=0;
//             for(uint_t k=0; k<cubic_root; ++k)
//                 for(uint_t j=0; j<cubic_root; ++j)
//                     for(uint_t i=0; i<cubic_root; ++i)
//                     {
//                         printf("ordered_quad_points_(%d,0)==cub_points(%d) ==> %f==%f\n",m,i, ordered_quad_points_(m,0) , cub_points_i(i,0));
//                         printf("ordered_quad_points_(%d,1)==cub_points(%d) ==> %f==%f\n", m,j,ordered_quad_points_(m,1) , cub_points_i(j,0));
//                         printf("ordered_quad_points_(%d,2)==cub_points(%d) ==> %f==%f\n\n", m,k,ordered_quad_points_(m,2) , cub_points_i(k,0));
//                         assert(ordered_quad_points_(m,0) == cub_points_i(i,0));
//                         assert(ordered_quad_points_(m,1) == cub_points_i(j,0));
//                         assert(ordered_quad_points_(m,2) == cub_points_i(k,0));
//                         ++m;
//                     }
// #endif

            // number of cubature points is correct
            // assert((gridtools::gt_pow<Dim>::apply(cubic_root) == ordered_quad_points_.dimension(0)));

            switch (op){
            case Intrepid::OPERATOR_VALUE :
            {
                Storage storage_ordered_(storage_);
                Intrepid::FieldContainer<gt::float_type> val_per_line_i(line_polynomial.getCardinality(), cubic_root);
                line_polynomial.getValues(val_per_line_i, cub_points_i, op);
                evaluate_polynomial< Dim, Order+1 >::apply(storage_ordered_, val_per_line_i, on_boundary);

                for(int_t i=0; i<storage_.dimension(0); ++i){
                    for(int_t q=0; q<storage_.dimension(1); ++q){
                        storage_(i,permutations[q])=storage_ordered_(i, q);
                    }
                }
                break;
            }
            case Intrepid::OPERATOR_GRAD :
            {
                Storage storage_ordered_(storage_);
                Intrepid::FieldContainer<gt::float_type> der_per_line_i(line_polynomial.getCardinality(), cubic_root, 1);
                Intrepid::FieldContainer<gt::float_type> val_per_line_i(line_polynomial.getCardinality(), cubic_root);
                line_polynomial.getValues(der_per_line_i, cub_points_i, Intrepid::OPERATOR_GRAD);
                line_polynomial.getValues(val_per_line_i, cub_points_i, Intrepid::OPERATOR_VALUE);
                evaluate_polynomial< Dim, Order+1 >::derivative(storage_ordered_, val_per_line_i, der_per_line_i, on_boundary);

                for(int_t i=0; i<storage_.dimension(0); ++i){
                    for(int_t q=0; q<storage_.dimension(1); ++q){
                        for(int_t d=0; d<storage_.dimension(2); ++d){
                            storage_(i,permutations[q],d)=storage_ordered_(i, q, d);
                        }
                    }
                }
                break;
            }
            default:
            {
                std::cout<<"Operator not supported"<<std::endl;
                assert(false);
     }
            }
        }
    };
}
