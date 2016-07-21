#pragma once

namespace gdl{
namespace functors{


    typedef gridtools::interval<gridtools::level<0,-1>, gridtools::level<1,-1> > x_interval;
    typedef gridtools::interval<gridtools::level<0,-2>, gridtools::level<1,1> > axis;



    // [update_jac]
    /**
        exactly the same as update_jac, just looping over the boundary faces. will need to be merged
        with the other version.
     */
    template<typename Geometry, enumtype::Shape S=Geometry::parent_shape >
    struct update_bd_jac{
        using cub=typename Geometry::cub;
        using geo_map=typename Geometry::geo_map;

        typedef gt::accessor<0, enumtype::in, gt::extent<0,0,0,0> , 5> const grid_points;
        typedef gt::accessor<1, enumtype::in, gt::extent<0,0,0,0> , 4> const dphi;
        typedef gt::accessor<2, enumtype::inout, gt::extent<0,0,0,0> , 7> jac;
        typedef boost::mpl::vector< grid_points, dphi, jac> arg_list;

        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_interval) {
            gt::dimension<4>::Index qp;
            gt::dimension<5>::Index dimx;
            gt::dimension<6>::Index dimy;
            gt::dimension<1>::Index i;
            gt::dimension<2>::Index j;
            gt::dimension<3>::Index k;

            uint_t const num_cub_points=eval.template get_storage_dims<1>(dphi());
            uint_t const basis_cardinality=eval.template get_storage_dims<0>(dphi());
            uint_t const n_faces_=eval.template get_storage_dims<6>(jac());

#ifndef __CUDACC__
            assert(num_cub_points==eval.template get_storage_dims<1>(dphi()));
#endif

            for(short_t face_=0; face_< n_faces_; ++face_)
            {
                //TODO dimensions should be generic
                for(short_t icoor=0; icoor< shape_property<Geometry::parent_shape>::dimension; ++icoor)
                {
                    for(short_t jcoor=0; jcoor< shape_property<S>::dimension; ++jcoor)
                    {
                        for(short_t iter_quad=0; iter_quad< num_cub_points; ++iter_quad)
                        {
                            eval( jac(dimx+icoor, dimy+jcoor, qp+iter_quad, gt::dimension<7>(face_) ) )=0.;
                            for (int_t iterNode=0; iterNode < basis_cardinality ; ++iterNode)
                            {//reduction/gather
                                eval( jac(dimx+icoor, dimy+jcoor, qp+iter_quad, gt::dimension<7>(face_)) ) += eval(grid_points(gt::dimension<4>(iterNode), gt::dimension<5>(icoor))  * !dphi(i+iterNode, j+iter_quad, k+jcoor, gt::dimension<4>(face_) ) );
                            }
                        }
                    }
                }
            }
        }
    };
    // [update_jac]



    // //! [det]
    // /** updates the values of the Jacobian matrix. The Jacobian matrix, component (i,j) in the quadrature point q, is computed given the geometric map discretization as \f$ J(i,j,q)=\sum_k\frac{\partial \phi_i(x_k,q)}{\partial x_j} x_k \f$
    //     where x_k are the points in the geometric element*/
    // template<typename BdGeometry>
    // struct bd_projection{
    //     using bd_cub=typename BdGeometry::cub;

    //     using jac =  gt::accessor<0, gt::extent<0,0,0,0> , 6> const;
    //     using normals =  gt::accessor<1, gt::extent<0,0,0,0> , 5> const;
    //     using jac_projected = gt::accessor<2, gt::extent<0,0,0,0> , 6>;
    //     using arg_list= boost::mpl::vector< jac, normals, jac_projected > ;

    //     template <typename Evaluation>
    //     GT_FUNCTION
    //     static void Do(Evaluation const & eval, x_interval) {
    //         dimension<4>::Index qp;
    //         dimension<5>::Index dimx;
    //         dimension<6>::Index dimy;

    //         uint_t const num_cub_points=eval.get_storage_dims(jac())[3];

    //         //"projection" on the tangent space:
    //         //J_{ij} - n_i n_k J_kj + n_i n_j
    //         for(short_t i=0; i< 3; ++i)
    //         {
    //             for(short_t j=0; j< 3; ++j)
    //             {
    //                 for(short_t q=0; q< num_cub_points; ++q)
    //                 {
    //                     float_type inner_product=0.;
    //                     for(short_t k=0; k< num_cub_points; ++k)
    //                     {
    //                         inner_product += eval(jac(dimx+i, dimy+j, qp+q))-
    //                             eval(normals(dimx+i, qp+q))*
    //                             eval(normals(dimx+k, qp+q))*
    //                             eval(jac(dimx+k, dimy+j, qp+q))+
    //                             eval(normals(dimx+i, qp+q))*
    //                             eval(normals(dimx+j, qp+q))//so that the matrix is not singular
    //                             ;
    //                     }
    //                     eval( jac_projected(dimx+i, dimy+j, qp+q) ) = inner_product;
    //                 }
    //             }
    //         }
    //     }
    // };



// [boundary integration]
/**
   This functor computes an integran over a boundary face
*/
    using namespace gridtools::expressions;
    template <typename FE, typename BoundaryCubature>
    struct bd_mass {
        using fe=FE;
        using bd_cub=BoundaryCubature;

        using jac_det=gt::accessor< 0, enumtype::in, gt::extent<0,0,0,0>, 5 >;
        using weights=gt::accessor< 1, enumtype::in, gt::extent<0,0,0,0>, 3 >;
        using phi_trace=gt::accessor< 2, enumtype::in, gt::extent<0,0,0,0>, 3 >;
        using psi_trace=gt::accessor< 3, enumtype::in, gt::extent<0,0,0,0>, 3 >;
        using out=gt::accessor< 4, enumtype::inout, gt::extent<0,0,0,0>, 6 >;

        using arg_list=boost::mpl::vector<jac_det, weights, phi_trace, psi_trace, out> ;

        /** @brief compute the integral on the boundary of a field times the normals

            note that we use here the traces of the basis functions, i.e. the basis functions
            evaluated on the quadrature points of the boundary faces.
        */
        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_interval) {
            gt::dimension<4>::Index quad;
            gt::dimension<4>::Index dofI;
            gt::dimension<5>::Index dofJ;

            uint_t const num_cub_points=eval.template get_storage_dims<3>(jac_det());
            uint_t const basis_cardinality = eval.template get_storage_dims<0>(phi_trace());
            uint_t const n_faces = eval.template get_storage_dims<4>(jac_det());


            for(short_t face_=0; face_<n_faces; ++face_) // current dof
            {
                //loop on the basis functions (interpolation in the quadrature point)
                //over the whole basis TODO: can be reduced
                for(short_t P_i=0; P_i<basis_cardinality; ++P_i) // current dof
                {
                    for(short_t P_j=0; P_j<basis_cardinality; ++P_j) // current dof
                    {
                        float_type partial_sum=0.;
                        for(ushort_t q_=0; q_<num_cub_points; ++q_){
                            // auto tmp1=eval(!phi_trace(P_i,q_,face_));
                            // auto tmp2=eval(!psi_trace(P_j, q_, face_));
                            // auto tmp3=eval(jac_det(quad+q_, gt::dimension<5>(face_)));
                            // auto tmp4=eval(!weights(q_));
                            // std::cout<<tmp1<<"\n";
                            // std::cout<<tmp2<<"\n";
                            // std::cout<<tmp3<<"\n";
                            // std::cout<<tmp4<<"\n";
                            partial_sum += eval(!phi_trace(P_i,q_,face_)*!psi_trace(P_j, q_, face_)*jac_det(quad+q_, gt::dimension<5>(face_)) * !weights(q_));
                        }
                        eval(out(dofI+P_i, dofJ+P_j, gt::dimension<6>(face_)))=partial_sum;
                    }
                }
            }
        }
    };
// [boundary integration]

// [boundary integration]
/**
   This functor computes an integran over a boundary face
*/
    using namespace gt::expressions;

    template <typename FE, typename BoundaryCubature>
    struct bd_mass_uv {
        using fe=FE;
        using bd_cub=BoundaryCubature;

        using jac_det=gt::accessor< 0, enumtype::in, gt::extent<0,0,0,0>, 5 >;
        using weights=gt::accessor< 1, enumtype::in, gt::extent<0,0,0,0>, 3 >;
        using phi_trace=gt::accessor< 2, enumtype::in, gt::extent<0,0,0,0>, 3 >;
        using psi_trace=gt::accessor< 3, enumtype::in, gt::extent<0,0,0,0>, 3 >;
        using out=gt::accessor< 4, enumtype::inout, gt::extent<0,0,0,0>, 6 >;

        using arg_list=boost::mpl::vector<jac_det, weights, phi_trace, psi_trace, out> ;

        /** @brief compute the integral on the boundary of a field times the normals

            note that we use here the traces of the basis functions, i.e. the basis functions
            evaluated on the quadrature points of the boundary faces.
        */
        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_interval) {
            gt::dimension<4>::Index quad;
            gt::dimension<4>::Index dofI;
            gt::dimension<5>::Index dofJ;

            uint_t const num_cub_points=eval.template get_storage_dims<3>(jac_det());
            uint_t const basis_cardinality = eval.template get_storage_dims<0>(phi_trace());
            uint_t const n_faces = eval.template get_storage_dims<4>(jac_det());


            for(short_t face_=0; face_<n_faces; ++face_) // current dof
            {
                short_t face_opposite_ =
                    face_==0?1
                    : face_==1?0
                    : face_==2?3
                    : face_==3?2
                    : face_==4?5
                    : face_==5?4
                    : -666;
                // loop on the basis functions (interpolation in the quadrature point)
                // over the whole basis TODO: can be reduced to the face dofs when the basis func.
                // are localized
                for(short_t P_i=0; P_i<basis_cardinality; ++P_i) // current dof
                {
                    for(short_t P_j=0; P_j<basis_cardinality; ++P_j) // current dof
                    {
                        float_type partial_sum=0.;
                        for(ushort_t q_=0; q_<num_cub_points; ++q_){
                            partial_sum += eval(!phi_trace(P_i,q_,face_)*!psi_trace(P_j, q_, face_opposite_)*jac_det(quad+q_, gt::dimension<5>(face_)) * !weights(q_));
                        }
                        //NOTE:
                        //we leave the local numeration on faces unchanged, so mass(i,i) does not
                        //correspond to 2 basis func. on the same point. Instead
                        //if i=point, j=opposite(point), then mass(i,j) is the "diagonal" entry
                        eval(out(dofI+P_i, dofJ+P_j, gt::dimension<6>(face_)))=partial_sum;
                    }
                }
            }
        }
    };
// [boundary integration]


    // //! [det]

    template<typename Geometry, ushort_t Codimensoin>
    struct measure;

    //! [measure]
    template<typename Geometry>
    struct measure<Geometry, 1>{
        using cub=typename Geometry::cub;

        using jac = gt::accessor<0, enumtype::in, gt::extent<0,0,0,0> , 7> const;
        using jac_det =  gt::accessor<1, enumtype::inout, gt::extent<0,0,0,0> , 5>;
        using arg_list= boost::mpl::vector< jac, jac_det > ;

        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_interval) {
            gt::dimension<4>::Index qp;
            gt::dimension<5>::Index dimx;
            gt::dimension<6>::Index dimy;

            //HARDCODED
            uint_t const num_faces=eval.template get_storage_dims<4>(jac_det());
            uint_t const num_cub_points=eval.template get_storage_dims<3>(jac_det());

            for(short_t face_=0; face_< num_faces; ++face_)
            {
#ifndef __CUDACC__
                gt::alias<jac, gt::dimension<7> > J(face_);
                gt::alias<jac_det, gt::dimension<5> > Jdet(face_);

                for(short_t q=0; q< num_cub_points; ++q)
                {
                    eval( Jdet(qp+q) )= eval(
                        (
                            J(        qp+q)*J(dimx+1, dimy+1, qp+q) +
                            J(dimx+1, qp+q)*J(dimx+2, dimy+1, qp+q) +
                            J(dimy+1, qp+q)*J(dimx+2,         qp+q) -
                            J(dimy+1, qp+q)*J(dimx+1,         qp+q) -
                            J(        qp+q)*J(dimx+2, dimy+1, qp+q) -
                            J(dimx+1, dimy+1, qp+q)*J(dimx+2,         qp+q)
                            )
                        );
                }
#else

                gt::dimension<7>::Index D7;
                gt::dimension<5>::Index D5;

                for(short_t q=0; q< num_cub_points; ++q)
                {
                    eval( jac_det(qp+q, D5+face_) )= eval(
                        (
                            jac(        qp+q, D7+face_)*jac(dimx+1, dimy+1, qp+q, D7+face_) +
                            jac(dimx+1, qp+q, D7+face_)*jac(dimx+2, dimy+1, qp+q, D7+face_) +
                            jac(dimy+1, qp+q, D7+face_)*jac(dimx+2,         qp+q, D7+face_) -
                            jac(dimy+1, qp+q, D7+face_)*jac(dimx+1,         qp+q, D7+face_) -
                            jac(        qp+q, D7+face_)*jac(dimx+2, dimy+1, qp+q, D7+face_) -
                            jac(dimx+1, dimy+1, qp+q, D7+face_)*jac(dimx+2,         qp+q, D7+face_)
                            )
                        );
                }
#endif

            }
        }
    };

    // template<typename Geometry, ushort_t SpaceDimension, ushort_t Codimensoin>
    // struct measure_impl;

    // template <typename Geometry, ushort_t Codimension>
    // using measure = measure_impl<Geometry, shape_property<Geometry::parent_shape>::dimension, Codimension>;

    // //avoid the code repetition with the functor above! (easily done)
    // template<typename Geometry>
    // struct measure_impl<Geometry, 2, 1>{
    //     using cub=typename Geometry::cub;

    //     using jac = gt::accessor<0, enumtype::in, gt::extent<0,0,0,0> , 7> const;
    //     using jac_det =  gt::accessor<1, enumtype::inout, gt::extent<0,0,0,0> , 5>;
    //     using arg_list= boost::mpl::vector< jac, jac_det > ;

    //     template <typename Evaluation>
    //     GT_FUNCTION
    //     static void Do(Evaluation const & eval, x_interval) {
    //         gt::dimension<4>::Index qp;
    //         gt::dimension<5>::Index dimx;
    //         gt::dimension<6>::Index dimy;

    //         uint_t const num_faces=eval.template get_storage_dims<4>(jac_det());
    //         uint_t const num_cub_points=eval.template get_storage_dims<3>(jac_det());

    //         for(short_t face_=0; face_< num_faces; ++face_)
    //         {
    //             alias<jac, gt::dimension<7> > J(face_);
    //             alias<jac_det, gt::dimension<5> > Jdet(face_);

    //             for(short_t q=0; q< num_cub_points; ++q)
    //             {
    //                 eval( Jdet(qp+q) )= eval(
    //                     (
    //                         J(        qp+q)*J(dimx+1, dimy+1, qp+q) - //probably wrong
    //                         J(dimy+1, qp+q)*J(dimx+1,         qp+q)
    //                         )
    //                     );
    //             }
    //         }
    //     }
    // };
    //! [measure]


    // [normals]

    /**
       Note: the reference normals are ordered as follows:
       index:                  .____.
       0 (0,-1,0)             /  0 /|
       1 (1,0,0)             .____. |5
       2 (0,1,0)             |    |1.          z
       3 (-1,0,0)           3|  4 |/       x__/
       4 (0,0,-1)            .____.           |
       5 (0,0,1)               2              y
     */
    template<typename BdGeometry>
    struct compute_face_normals{
        using bd_cub=typename BdGeometry::cub;
        static const auto parent_shape=BdGeometry::parent_shape;

        using jac=gt::accessor< 0, enumtype::in, gt::extent<>, 7 >;
        using ref_normals=gt::accessor< 1, enumtype::in, gt::extent<>, 3 >;
        using normals=gt::accessor< 2, enumtype::inout, gt::extent<>, 6 >;
        using arg_list=boost::mpl::vector<jac, ref_normals, normals> ;

        /** @brief compute the normal vectors in the face quadrature points

            compute the map of the tangent vectors, and take their vector product
            (works also for non-conformal maps)
        */
        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_interval) {
            gt::dimension<1>::Index i;
            gt::dimension<2>::Index j;
            gt::dimension<3>::Index k;
            gt::dimension<4>::Index quad;
            gt::dimension<5>::Index dimI;
            gt::dimension<6>::Index dimJ;
            gt::dimension<7>::Index f;
            uint_t const num_cub_points=eval.template get_storage_dims<3>(jac());
            uint_t const num_faces=eval.template get_storage_dims<6>(jac());

            for(ushort_t face_=0; face_<num_faces; ++face_){
                for(ushort_t q_=0; q_<num_cub_points; ++q_){
                    //TODO: hardcoded 3
                    for(ushort_t i_=0; i_<3; ++i_){
                        double product = 0.;
                        for(ushort_t j_=0; j_<3; ++j_){
                            // auto tmp1=eval(jac(quad+q_, dimI+i_, dimJ+j_, f+face_));
                            // auto tmp2=eval(!ref_normals(j_,face_));
                            // std::cout<<"face: "<<face_ <<" => "tmp1<<" * "<<tmp2<<"\n";
                            product += eval(jac(quad+q_, dimI+i_, dimJ+j_, f+face_)) * eval(!ref_normals(j_,face_));
                        }
                        eval(normals(quad+q_, dimI+i_, dimJ+face_)) = product;
                    }
                }
            }
        }
    };


    /**
       @brief integrate on face
     */
    template<typename BdGeometry>
    struct bd_integrate{
        using bd_cub=typename BdGeometry::cub;
        static const auto parent_shape=BdGeometry::parent_shape;

        using phi_trace=gt::accessor< 0, enumtype::in, gt::extent<>, 3 >;
        using jac_det=gt::accessor< 1, enumtype::in, gt::extent<>, 5 >;
        using weights=gt::accessor< 2, enumtype::in, gt::extent<>, 3 >;
        using in=gt::accessor< 3, enumtype::in, gt::extent<>, 6 >;
        using out=gt::accessor< 4, enumtype::inout, gt::extent<>, 6 >;
        using arg_list=boost::mpl::vector<phi_trace, jac_det, weights, in, out> ;

        /** @brief compute the integral of a vector

        */
        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_interval) {
            gt::dimension<1>::Index i;
            gt::dimension<2>::Index j;
            gt::dimension<3>::Index k;
            gt::dimension<4>::Index quad;
            gt::dimension<4>::Index dimI;
            gt::dimension<5>::Index sdim;
            gt::dimension<6>::Index f;

            uint_t const basis_cardinality = eval.template get_storage_dims<0>(phi_trace());
            uint_t const num_cub_points=eval.template get_storage_dims<1>(phi_trace());
            uint_t const num_faces=eval.template get_storage_dims<2>(phi_trace());
// #ifndef __CUDACC__
//             std::cout<<eval.template get_storage_dims<1>(phi_trace())<<std::endl;
//             assert(num_cub_points == eval.template get_storage_dims<1>(phi_trace()));
// #endif

            for(ushort_t face_=0; face_<num_faces; ++face_){
                for(ushort_t q_=0; q_<num_cub_points; ++q_){
                    for(ushort_t i_=0; i_<2; ++i_){
                        for(ushort_t dof_=0; dof_<basis_cardinality; ++dof_){

                            eval(out(dimI+dof_, sdim+i_, f+face_)) +=  eval(in(quad+q_, sdim+i_, f+face_) * !phi_trace(dof_,q_, face_)
                                                                            * jac_det(quad+q_, gt::dimension<5>(face_)) *
                                                                                      !weights(q_));
                        }
                    }
                }
            }
        }
    };

    // [normals]

    // template<typename BdGeometry, ushort_t faceID>
    // struct map_vectors{
    //     using bd_cub=typename BdGeometry::cub;
    //     static const auto parent_shape=BdGeometry::parent_shape;

    //     using jac=gt::accessor< 0, gt::extent<>, 6 >;
    //     using normals=gt::accessor< 1, gt::extent<>, 5 >;
    //     using arg_list=boost::mpl::vector<jac, normals> ;

    //     map_vectors()
    //         {}

    //     /** @brief compute the normal vectors in the face quadrature points

    //         compute the map of the tangent vectors, and take their vector product
    //         (works also for non-conformal maps)
    //     */
    //     template <typename Evaluation>
    //     GT_FUNCTION
    //     static void Do(Evaluation const & eval, x_interval) {
    //         x::Index i;
    //         y::Index j;
    //         z::Index k;
    //         gt::dimension<4>::Index quad;
    //         gt::dimension<5>::Index dimI;
    //         gt::dimension<6>::Index dimJ;

    //         uint_t const num_cub_points=eval.get_storage_dims(jac())[3];

    //         array<double, 3> tg_u;
    //         array<double, 3> tg_v;

    //         for(ushort_t q_=0; q_<num_cub_points; ++q_){
    //             for(ushort_t i_=0; i_<3; ++i_){
    //                 for(ushort_t j_=0; j_<3; ++j_){
    //                     tg_u[j_]=shape_property<parent_shape>::template tangent_u<faceID>::value[i_]*eval(jac(quad+q_, dimI+i_, dimJ+j_));
    //                     tg_v[j_]=shape_property<parent_shape>::template tangent_v<faceID>::value[i_]*eval(jac(quad+q_, dimI+i_, dimJ+j_));
    //                 }
    //             }
    //             array<double, 3> normal(vec_product(tg_u, tg_v));
    //             for(ushort_t j_=0; j_<3; ++j_){
    //                 eval(normals(quad+q_, dimJ+j_))=normal[j_];
    //             }
    //         }
    //     }
    // };
    // // [normals]



}//namespace functors
}//namespace gdl
