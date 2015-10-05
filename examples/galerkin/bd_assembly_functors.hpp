#pragma once

namespace functors{


    typedef gridtools::interval<gridtools::level<0,-1>, gridtools::level<1,-1> > x_interval;
    typedef gridtools::interval<gridtools::level<0,-2>, gridtools::level<1,1> > axis;

    //! [det]
    /** updates the values of the Jacobian matrix. The Jacobian matrix, component (i,j) in the quadrature point q, is computed given the geometric map discretization as \f$ J(i,j,q)=\sum_k\frac{\partial \phi_i(x_k,q)}{\partial x_j} x_k \f$
        where x_k are the points in the geometric element*/
    template<typename BdGeometry>
    struct bd_projection{
        using bd_cub=typename BdGeometry::cub;

        using jac_projected = accessor<0, range<0,0,0,0> , 6> const;
        using jac =  accessor<1, range<0,0,0,0> , 6>;
        using normals =  accessor<2, range<0,0,0,0> , 5>;
        using arg_list= boost::mpl::vector< jac_projected, jac, normals > ;

        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_interval) {
            dimension<4>::Index qp;
            dimension<5>::Index dimx;
            dimension<6>::Index dimy;

            uint_t const num_cub_points=eval.get().get_storage_dims(jac())[3];

            //"projection" on the tangent space:
            //J_{ij} - n_i n_k J_kj + n_i n_j
            for(short_t i=0; i< 3; ++i)
            {
                for(short_t j=0; j< 3; ++j)
                {
                    for(short_t q=0; q< num_cub_points; ++q)
                    {
                        float_type inner_product=0.;
                        for(short_t k=0; k< num_cub_points; ++k)
                        {
                            inner_product += eval(jac(dimx+i, dimy+j, qp+q))-
                                eval(normals(dimx+i, qp+q))*
                                eval(normals(dimx+k, qp+q))*
                                eval(jac(dimx+k, dimy+j, qp+q))+
                                eval(normals(dimx+i, qp+q))*
                                eval(normals(dimx+j, qp+q))//so that the matrix is not singular
                                ;
                        }
                        eval( jac_projected(dimx+i, dimy+j, qp+q) ) = inner_product;
                    }
                }
            }
        }
    };


//! [det]

    template<typename Geometry, ushort_t Codimensoin>
    struct measure;

    //! [measure]
    template<typename Geometry>
    struct measure<Geometry, 2>{
        using cub=typename Geometry::cub;

        using jac = accessor<0, range<0,0,0,0> , 6> const;
        using jac_det =  accessor<1, range<0,0,0,0> , 4>;
        using arg_list= boost::mpl::vector< jac, jac_det > ;

        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_interval) {
            dimension<4>::Index qp;
            dimension<5>::Index dimx;
            dimension<6>::Index dimy;

            uint_t const num_cub_points=eval.get().get_storage_dims(jac_det())[3];

            for(short_t q=0; q< num_cub_points; ++q)
            {
                eval( jac_det(qp+q) )= eval(
                    (
                        jac(        qp+q)*jac(dimx+1, dimy+1, qp+q) +
                        jac(dimx+1, qp+q)*jac(dimx+2, dimy+1, qp+q) +
                        jac(dimy+1, qp+q)*jac(dimx+2,         qp+q) -
                        jac(dimy+1, qp+q)*jac(dimx+1,         qp+q) -
                        jac(        qp+q)*jac(dimx+2, dimy+1, qp+q) -
                        jac(dimx+1, dimy+1, qp+q)*jac(dimx+2,         qp+q)
                        )
                    );
            }
        }
    };
    //! [measure]


    // [normals]
    template<typename BdGeometry>
    struct compute_face_normals{
        using bd_cub=typename BdGeometry::cub;
        static const auto parent_shape=BdGeometry::parent_shape;

        using jac=accessor< 0, range<0,0,0,0>, 6 >;
        using ref_normals=accessor< 1, range<0,0,0,0>, 3 >;
        using normals=accessor< 2, range<0,0,0,0>, 5 >;
        using arg_list=boost::mpl::vector<jac, ref_normals, normals> ;

        /** @brief compute the normal vectors in the face quadrature points

            compute the map of the tangent vectors, and take their vector product
            (works also for non-conformal maps)
        */
        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_interval) {
            x::Index i;
            y::Index j;
            z::Index k;
            dimension<4>::Index quad;
            dimension<5>::Index dimI;
            dimension<6>::Index dimJ;
            uint_t const num_cub_points=eval.get().get_storage_dims(jac())[3];

            for(ushort_t q_=0; q_<num_cub_points; ++q_){
                for(ushort_t i_=0; i_<3; ++i_){
                    double product = 0.;
                    for(ushort_t j_=0; j_<3; ++j_){
                        product += eval(jac(quad+q_, dimI+i_, dimJ+j_)) * eval(!ref_normals(j_,0,0));
                    }
                    eval(normals(quad+q_, dimI+i_)) = product;
                }
            }
        }
    };
    // [normals]

    template<typename BdGeometry, ushort_t faceID>
    struct map_vectors{
        using bd_cub=typename BdGeometry::cub;
        static const auto parent_shape=BdGeometry::parent_shape;

        using jac=accessor< 0, range<0,0,0,0>, 6 >;
        using normals=accessor< 1, range<0,0,0,0>, 5 >;
        using arg_list=boost::mpl::vector<jac, normals> ;

        map_vectors()
            {}

        /** @brief compute the normal vectors in the face quadrature points

            compute the map of the tangent vectors, and take their vector product
            (works also for non-conformal maps)
        */
        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_interval) {
            x::Index i;
            y::Index j;
            z::Index k;
            dimension<4>::Index quad;
            dimension<5>::Index dimI;
            dimension<6>::Index dimJ;

            uint_t const num_cub_points=eval.get().get_storage_dims(jac())[3];

            array<double, 3> tg_u;
            array<double, 3> tg_v;

            for(ushort_t q_=0; q_<num_cub_points; ++q_){
                for(ushort_t i_=0; i_<3; ++i_){
                    for(ushort_t j_=0; j_<3; ++j_){
                        tg_u[j_]=shape_property<parent_shape>::template tangent_u<faceID>::value[i_]*eval(jac(quad+q_, dimI+i_, dimJ+j_));
                        tg_v[j_]=shape_property<parent_shape>::template tangent_v<faceID>::value[i_]*eval(jac(quad+q_, dimI+i_, dimJ+j_));
                    }
                }
                array<double, 3> normal(vec_product(tg_u, tg_v));
                for(ushort_t j_=0; j_<3; ++j_){
                    eval(normals(quad+q_, dimJ+j_))=normal[j_];
                }
            }
        }
    };
    // [normals]



}//namespace functors
