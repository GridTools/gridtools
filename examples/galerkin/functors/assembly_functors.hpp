#pragma once

#include "../utils/indexing.hpp"

namespace gdl{
    namespace functors{

    //watchout: propagating namespace
    // using namespace gridtools;
    using namespace gridtools::expressions;

    typedef gridtools::interval<gridtools::level<0,-1>, gridtools::level<1,-1> > x_interval;
    typedef gridtools::interval<gridtools::level<0,-2>, gridtools::level<1,1> > axis;

    // [update_jac]
    /** updates the values of the Jacobian matrix. The Jacobian matrix, component (i,j) in the quadrature point q, is computed given the geometric map discretization as \f$ J(i,j,q)=\sum_k\frac{\partial \phi_i(x_k,q)}{\partial x_j} x_k \f$
        where x_k are the points in the geometric element*/
    template<typename Geometry, enumtype::Shape S=Geometry::parent_shape >
    struct update_jac{
        using cub=typename Geometry::cub;
        using geo_map=typename Geometry::geo_map;

        typedef gt::accessor<0, enumtype::in, gt::extent<0,0,0,0> , 5> const grid_points;
        typedef gt::accessor<1, enumtype::in, gt::extent<0,0,0,0> , 3> const dphi;
        typedef gt::accessor<2, enumtype::inout, gt::extent<0,0,0,0> , 6> jac;
        typedef boost::mpl::vector< grid_points, dphi, jac> arg_list;

        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_interval) {
            gt::dimension<4> qp;
            gt::dimension<5> dimx;
            gt::dimension<6> dimy;
            gt::dimension<1> i;
            gt::dimension<2> j;
            gt::dimension<3> k;

            uint_t const num_cub_points=eval.get().template get_storage_dim<1>(dphi());
            uint_t const basis_cardinality=eval.get().template get_storage_dim<0>(dphi());

#ifndef __CUDACC__
            assert(num_cub_points==cub::numCubPoints());
#endif
            //TODO dimensions should be generic
            for(short_t icoor=0; icoor< shape_property<Geometry::parent_shape>::dimension; ++icoor)
            {
                for(short_t jcoor=0; jcoor< shape_property<S>::dimension; ++jcoor)
                {
                    for(short_t iter_quad=0; iter_quad< num_cub_points; ++iter_quad)
                    {
                        eval( jac(dimx+icoor, dimy+jcoor, qp+iter_quad) )=0.;
                                for (int_t iterNode=0; iterNode < basis_cardinality ; ++iterNode)
                                {//reduction/gather
                                    eval( jac(dimx+icoor, dimy+jcoor, qp+iter_quad) ) +=
                                        eval(grid_points(gt::dimension<4>(iterNode), gt::dimension<5>(icoor)) * !dphi(i+iterNode, j+iter_quad, k+jcoor) );
                                }
                    }
                }
            }
        }
    };
    // [update_jac]


    //! [det]
    /** updates the values of the Jacobian matrix. The Jacobian matrix, component (i,j) in the quadrature point q, is computed given the geometric map discretization as \f$ J(i,j,q)=\sum_k\frac{\partial \phi_i(x_k,q)}{\partial x_j} x_k \f$
        where x_k are the points in the geometric element*/
    template<typename Geometry, ushort_t Dim=Geometry::geo_map::space_dim()>
    struct det_impl;

        template<typename Geometry>
        struct det
        {
            using jac = gt::accessor<0, enumtype::in, gt::extent<0,0,0,0> , 6> const;
            using jac_det =  gt::accessor<1, enumtype::inout, gt::extent<0,0,0,0> , 4>;
            using arg_list= boost::mpl::vector< jac, jac_det > ;

            template <typename Evaluation>
            GT_FUNCTION
            static void Do(Evaluation const & eval, x_interval) {

                gt::dimension<4> qp;
                gt::dimension<5> dimx;
                gt::dimension<6> dimy;
                uint_t const num_cub_points=eval.get().template get_storage_dim<3>(jac());

#ifdef __CUDACC__
                assert(num_cub_points==cub::numCubPoints());
#endif

                for(short_t q=0; q< num_cub_points; ++q)
                {
                    det_impl<Geometry>::DoCompute(eval,qp,dimx,dimy,q);
                }
        }
	};


    template<typename Geometry>// TODO: number of dimensions can be derived from Geometry
    struct det_impl<Geometry,3> : public det<Geometry>
    {
        using cub=typename Geometry::cub;
        using super=det<Geometry>;
        using jacobian= typename super::jac;
        using jacobian_det=typename super::jac_det;

        template <typename Evaluation>
        GT_FUNCTION
        static void DoCompute(Evaluation const & eval, const gt::dimension<4>& i_qp, const gt::dimension<5>::Index& i_dimx, const gt::dimension<6>::Index& i_dimy, const short_t i_q) {
	  eval( jacobian_det(i_qp+i_q) )= eval((
				jacobian(        i_qp+i_q)*jacobian(i_dimx+1, i_dimy+1, i_qp+i_q)*jacobian(i_dimx+2, i_dimy+2, i_qp+i_q) +
				jacobian(i_dimx+1, i_qp+i_q)*jacobian(i_dimx+2, i_dimy+1, i_qp+i_q)*jacobian(i_dimy+2,         i_qp+i_q) +
				jacobian(i_dimy+1, i_qp+i_q)*jacobian(i_dimx+1, i_dimy+2, i_qp+i_q)*jacobian(i_dimx+2,         i_qp+i_q) -
				jacobian(i_dimy+1, i_qp+i_q)*jacobian(i_dimx+1,         i_qp+i_q)*jacobian(i_dimx+2, i_dimy+2, i_qp+i_q) -
				jacobian(        i_qp+i_q)*jacobian(i_dimx+2, i_dimy+1, i_qp+i_q)*jacobian(i_dimx+1, i_dimy+2, i_qp+i_q) -
				jacobian(i_dimy+2, i_qp+i_q)*jacobian(i_dimx+1, i_dimy+1, i_qp+i_q)*jacobian(i_dimx+2,         i_qp+i_q)));
        }
    };


    template<typename Geometry>
    struct det_impl<Geometry,2> : public det<Geometry>
    {
        using cub=typename Geometry::cub;
        using super=det<Geometry>;
        using jacobian=typename super::jac;
        using jacobian_det=typename super::jac_det;

        template <typename Evaluation>
        GT_FUNCTION
        static void DoCompute(Evaluation const & eval, const gt::dimension<4>& i_qp, const gt::dimension<5>::Index& i_dimx, const gt::dimension<6>::Index& i_dimy, const short_t i_q) {
            eval( jacobian_det(i_qp+i_q) )= eval((
		     jacobian(        i_qp+i_q)*jacobian(i_dimx+1, i_dimy+1, i_qp+i_q) -
		     jacobian(i_dimx+1, i_qp+i_q)*jacobian(i_dimy+1, i_qp+i_q)));
        }

    };
    //! [det]


    //! [inv]

    template <typename Geometry,ushort_t Dim=Geometry::geo_map::space_dim()>
    struct inv_impl;

    template <typename Geometry>
    struct inv
    {
        using cub=typename Geometry::cub;

        //![arguments_inv]
        /**The input arguments to this functors are the matrix and its determinant. */
        using jac      = gt::accessor<0, enumtype::in, gt::extent<0,0,0,0> , 6> const ;
        using jac_det  = gt::accessor<1, enumtype::in, gt::extent<0,0,0,0> , 4> const ;
        using jac_inv  = gt::accessor<2, enumtype::inout, gt::extent<0,0,0,0> , 6> ;
        using arg_list = boost::mpl::vector< jac, jac_det, jac_inv>;
        //![arguments_inv]

    	template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_interval) {
        	inv_impl<Geometry>::DoCompute(eval);
        }
    };

#ifndef __CUDACC__
    template <typename Geometry>
    struct inv_impl<Geometry,3> : public inv<Geometry>
    {
    	using super=inv<Geometry>;

        using jac      = typename super::jac ;
        using jac_det  = typename super::jac_det ;
        using jac_inv  = typename super::jac_inv ;

        template <typename Evaluation>
        GT_FUNCTION
        static void DoCompute(Evaluation const & eval) {

            gt::dimension<4> qp;
            using dimx=gt::dimension<5>;
            using dimy=gt::dimension<6>;
            dimx X;
            dimy Y;
            uint_t const num_cub_points=eval.get().template get_storage_dim<3>(jac());

            typedef typename Geometry::cub cub;
            assert(num_cub_points==cub::numCubPoints());

//! [aliases]
            using a_=typename gt::alias<jac, dimy, dimx>::template set<0,0>;
            using b_=typename gt::alias<jac, dimy, dimx>::template set<0,1>;
            using c_=typename gt::alias<jac, dimy, dimx>::template set<0,2>;
            using d_=typename gt::alias<jac, dimy, dimx>::template set<1,0>;
            using e_=typename gt::alias<jac, dimy, dimx>::template set<1,1>;
            using f_=typename gt::alias<jac, dimy, dimx>::template set<1,2>;
            using g_=typename gt::alias<jac, dimy, dimx>::template set<2,0>;
            using h_=typename gt::alias<jac, dimy, dimx>::template set<2,1>;
            using i_=typename gt::alias<jac, dimy, dimx>::template set<2,2>;
//! [aliases]
            // eval( jac(dimx+icoor, dimy+jcoor, qp+iter_quad) )=0.;
            for(short_t q=0; q< num_cub_points; ++q)
            {
//                gt::alias<a_, gt::dimension<4> > a(q);
//                gt::alias<b_, gt::dimension<4> > b(q);
//                gt::alias<c_, gt::dimension<4> > c(q);
//                gt::alias<d_, gt::dimension<4> > d(q);
//                gt::alias<e_, gt::dimension<4> > e(q);
//                gt::alias<f_, gt::dimension<4> > f(q);
//                gt::alias<g_, gt::dimension<4> > g(q);
//                gt::alias<h_, gt::dimension<4> > h(q);
//                gt::alias<i_, gt::dimension<4> > i(q);
//
//                assert(eval(a()) == eval(jac(qp+q)));
//                assert(eval(b()) == eval(jac(qp+q, X+1)));
//                assert(eval(c()) == eval(jac(qp+q, X+2)));
//                assert(eval(d()) == eval(jac(qp+q, Y+1)));

                //! std::cout << "JACOBIAN: "<<std::endl;
                //! std::cout<<eval(a())<<" "<<eval(b())<<" "<<eval(c())<<std::endl;
                //! std::cout<<eval(d())<<" "<<eval(e())<<" "<<eval(f())<<std::endl;
                //! std::cout<<eval(g())<<" "<<eval(h())<<" "<<eval(i())<<std::endl;
//
//                eval( jac_inv(qp+q) )           = eval( ( e()*i() - f()*h())/jac_det(qp+q));
//                eval( jac_inv(X+1, qp+q) )      = eval( ( f()*g() - d()*i())/jac_det(qp+q));
//                eval( jac_inv(X+2, qp+q) )      = eval( ( d()*h() - e()*g())/jac_det(qp+q));
//                eval( jac_inv(Y+1, qp+q) )      = eval( ( c()*h() - b()*i())/jac_det(qp+q));
//                eval( jac_inv(Y+1, X+1, qp+q) ) = eval( ( a()*i() - c()*g())/jac_det(qp+q));
//                eval( jac_inv(Y+1, X+2, qp+q) ) = eval( ( b()*g() - a()*h())/jac_det(qp+q));
//                eval( jac_inv(Y+2, qp+q) )      = eval( ( b()*f() - c()*e())/jac_det(qp+q));
//                eval( jac_inv(Y+2, X+1, qp+q) ) = eval( ( c()*d() - a()*f())/jac_det(qp+q));
//                eval( jac_inv(Y+2, X+2, qp+q) ) = eval( ( a()*e() - b()*d())/jac_det(qp+q));
//
//                //! std::cout << "JACOBIAN INVERSE: "<<std::endl;
//                //! std::cout<<eval(jac_inv(qp+q))<<" "<<eval(jac_inv(qp+q, X+1))<<" "<<eval(jac_inv(qp+q, X+2))<<std::endl;
//                //! std::cout<<eval(jac_inv(qp+q, Y+1))<<" "<<eval(jac_inv(qp+q, X+1, Y+1))<<" "<<eval(jac_inv(qp+q, X+2, Y+1))<<std::endl;
//                //! std::cout<<eval(jac_inv(qp+q, Y+2))<<" "<<eval(jac_inv(qp+q, X+1, Y+2))<<" "<<eval(jac_inv(qp+q, X+2, Y+2))<<std::endl;

                eval( jac_inv(qp+q) )           = eval( ( jac(qp+q, X+1, Y+1)*jac(qp+q, X+2,Y+2)) - jac(qp+q, X+2, Y+1)*jac(qp+q, X+1,Y+2)/jac_det(qp+q));
                eval( jac_inv(X+1, qp+q) )      = eval( ( jac(qp+q, X+2, Y+1)*jac(qp+q, Y+2)) - jac(qp+q, Y+1)*jac(qp+q, X+2,Y+2)/jac_det(qp+q));
                eval( jac_inv(X+2, qp+q) )      = eval( ( jac(qp+q, Y+1)*jac(qp+q, X+1,Y+2)) - jac(qp+q, X+1, Y+1)*jac(qp+q, Y+2)/jac_det(qp+q));
                eval( jac_inv(Y+1, qp+q) )      = eval( ( jac(qp+q, X+2)*jac(qp+q, X+1,Y+2)) - jac(qp+q, X+1)*jac(qp+q, X+2,Y+2)/jac_det(qp+q));
                eval( jac_inv(Y+1, X+1, qp+q) ) = eval( ( jac(qp+q)*jac(qp+q, X+2,Y+2)) - jac(qp+q, X+2)*jac(qp+q, Y+2)/jac_det(qp+q));
                eval( jac_inv(Y+1, X+2, qp+q) ) = eval( ( jac(qp+q, X+1)*jac(qp+q, Y+2)) - jac(qp+q)*jac(qp+q, X+1,Y+2)/jac_det(qp+q));
                eval( jac_inv(Y+2, qp+q) )      = eval( ( jac(qp+q, X+1)*jac(qp+q, X+2, Y+1)) - jac(qp+q, X+2)*jac(qp+q, X+1, Y+1)/jac_det(qp+q));
                eval( jac_inv(Y+2, X+1, qp+q) ) = eval( ( jac(qp+q, X+2)*jac(qp+q, Y+1)) - jac(qp+q)*jac(qp+q, X+2, Y+1)/jac_det(qp+q));
                eval( jac_inv(Y+2, X+2, qp+q) ) = eval( ( jac(qp+q)*jac(qp+q, X+1, Y+1)) - jac(qp+q, X+1)*jac(qp+q, Y+1)/jac_det(qp+q));

            }

        }
    };

    template <typename Geometry>
    struct inv_impl<Geometry,2> : public inv<Geometry>
    {
    	using super=inv<Geometry>;

        using jac      = typename super::jac ;
        using jac_det  = typename super::jac_det ;
        using jac_inv  = typename super::jac_inv ;

        template <typename Evaluation>
        GT_FUNCTION
        static void DoCompute(Evaluation const & eval) {
            gt::dimension<4> qp;
            using dimx=gt::dimension<5>;
            using dimy=gt::dimension<6>;
            dimx X;
            dimy Y;
            uint_t const num_cub_points=eval.get().template get_storage_dim<3>(jac());

#ifdef __CUDACC__
            assert(num_cub_points==cub::numCubPoints());
#endif

//! [aliases]
            using a_= typename gt::alias<jac, dimy, dimx>::template set<0,0>;
            using b_= typename gt::alias<jac, dimy, dimx>::template set<0,1>;
            using c_= typename gt::alias<jac, dimy, dimx>::template set<1,0>;
            using d_= typename gt::alias<jac, dimy, dimx>::template set<1,1>;
//! [aliases]
            // eval( jac(dimx+icoor, dimy+jcoor, qp+iter_quad) )=0.;
            for(short_t q=0; q< num_cub_points; ++q)
            {
                gt::alias<a_, gt::dimension<4> > a(q);
                gt::alias<b_, gt::dimension<4> > b(q);
                gt::alias<c_, gt::dimension<4> > c(q);
                gt::alias<d_, gt::dimension<4> > d(q);

                assert(eval(a()) == eval(jac(qp+q)));
                assert(eval(b()) == eval(jac(qp+q, X+1)));
                assert(eval(c()) == eval(jac(qp+q, Y+1)));
                assert(eval(d()) == eval(jac(qp+q, X+1,Y+1)));

                eval( jac_inv(qp+q) )           = eval( d()/jac_det(qp+q) );
                eval( jac_inv(X+1, qp+q) )      = -eval( c()/jac_det(qp+q) );
                eval( jac_inv(Y+1, qp+q) )      = -eval( b()/jac_det(qp+q) );
                eval( jac_inv(Y+1, X+1, qp+q) ) = eval( a()/jac_det(qp+q) );

            }
        }
    };

#else //__CUDACC__

    template <typename Geometry>
    struct inv_impl<Geometry,3> : public inv<Geometry>
    {
        using super=inv<Geometry>;

        using jac      = typename super::jac ;
        using jac_det  = typename super::jac_det ;
        using jac_inv  = typename super::jac_inv ;

        template <typename Evaluation>
        GT_FUNCTION
        static void DoCompute(Evaluation const & eval) {

            gt::dimension<4> qp;
            using dimx=gt::dimension<5>;
            using dimy=gt::dimension<6>;
            dimx X;
            dimy Y;
            uint_t const num_cub_points=eval.get().template get_storage_dim<3>(jac());


            assert(num_cub_points==cub::numCubPoints());

            for(short_t q=0; q< num_cub_points; ++q)
            {

                eval( jac_inv(qp+q) )           = eval( ( jac(qp+q, X+1, Y+1)*jac(qp+q, X+2,Y+2)) - jac(qp+q, X+2, Y+1)*jac(qp+q, X+1,Y+2)/jac_det(qp+q));
                eval( jac_inv(X+1, qp+q) )      = eval( ( jac(qp+q, X+2, Y+1)*jac(qp+q, Y+2)) - jac(qp+q, Y+1)*jac(qp+q, X+2,Y+2)/jac_det(qp+q));
                eval( jac_inv(X+2, qp+q) )      = eval( ( jac(qp+q, Y+1)*jac(qp+q, X+1,Y+2)) - jac(qp+q, X+1, Y+1)*jac(qp+q, Y+2)/jac_det(qp+q));
                eval( jac_inv(Y+1, qp+q) )      = eval( ( jac(qp+q, X+2)*jac(qp+q, X+1,Y+2)) - jac(qp+q, X+1)*jac(qp+q, X+2,Y+2)/jac_det(qp+q));
                eval( jac_inv(Y+1, X+1, qp+q) ) = eval( ( jac(qp+q)*jac(qp+q, X+2,Y+2)) - jac(qp+q, X+2)*jac(qp+q, Y+2)/jac_det(qp+q));
                eval( jac_inv(Y+1, X+2, qp+q) ) = eval( ( jac(qp+q, X+1)*jac(qp+q, Y+2)) - jac(qp+q)*jac(qp+q, X+1,Y+2)/jac_det(qp+q));
                eval( jac_inv(Y+2, qp+q) )      = eval( ( jac(qp+q, X+1)*jac(qp+q, X+2, Y+1)) - jac(qp+q, X+2)*jac(qp+q, X+1, Y+1)/jac_det(qp+q));
                eval( jac_inv(Y+2, X+1, qp+q) ) = eval( ( jac(qp+q, X+2)*jac(qp+q, Y+1)) - jac(qp+q)*jac(qp+q, X+2, Y+1)/jac_det(qp+q));
                eval( jac_inv(Y+2, X+2, qp+q) ) = eval( ( jac(qp+q)*jac(qp+q, X+1, Y+1)) - jac(qp+q, X+1)*jac(qp+q, Y+1)/jac_det(qp+q));
            }

        }
    };

// [inv]
#endif //__CUDACC__


    // [assemble]
    /**
       @class functor assembling a vector

       Given a quantity defined on the grid it loops over the lower boundary dofs of the current element
       (i.e. the boundary corresponding to a lower index) and computes an operation (\tparam Operator) between the value at the boundary
       and the corresponding one in the neighboring element.
     */
    template<typename Geometry>
    struct assemble {

        using geo_map=typename Geometry::geo_map;

        using in2=gt::accessor<0, enumtype::in, gt::extent<> , 4>;
        using out=gt::accessor<1, enumtype::inout, gt::extent<> , 4> ;
        using arg_list=boost::mpl::vector<in2, out> ;

        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_interval) {
            gt::dimension<1> i;
            gt::dimension<2> j;
            gt::dimension<3> k;
            gt::dimension<4> row;


            //hypothesis here: the cardinaxlity is order^3 (isotropic 3D tensor product element)
#ifdef __CUDACC__
#ifdef NDEBUG
            constexpr
#endif
                gt::meta_storage_base<static_int<__COUNTER__>,gt::layout_map<2,1,0>,false> indexing{static_int<1+1>(), static_int<1+1>(), static_int<1+1>()};
#else
#ifdef NDEBUG
            constexpr
#endif
                gt::meta_storage_base<static_int<__COUNTER__>,gt::layout_map<2,1,0>,false> indexing{Geometry::geo_map::order+1, Geometry::geo_map::order+1, Geometry::geo_map::order+1};

#endif

            uint_t N1 = indexing.template dim<0>()-1;
            uint_t N2 = indexing.template dim<1>()-1;
            uint_t N3 = indexing.template dim<2>()-1;

            // for all dofs in a boundary face (supposing that the dofs per face are the same)
            // setting dofs internal to the faces
            for(short_t I=1; I<indexing.template dim<0>()-1; I++)
                for(short_t J=1; J<indexing.template dim<1>()-1; J++)
                {

                    //for each (3) faces
                    auto dof_x=indexing.index(0, (int)I, (int)J);
                    auto dof_xx=indexing.index(indexing.template dim<0>()-1, I, J);
                    auto dof_y=indexing.index(I, 0, J);
                    auto dof_yy=indexing.index(I, indexing.template dim<1>()-1, J);
                    auto dof_z=indexing.index(I, J, 0);
                    auto dof_zz=indexing.index(I, J, indexing.template dim<2>()-1);

                    // z=0 face
                    //sum the contribution from elem i-1 on the opposite face
                    eval(out(row+dof_x)) +=  eval(in2(k-1, row+dof_xx));

                    //sum the contribution from elem j-1 on the opposite face
                    eval(out(row+dof_y)) +=  eval(in2(k-1, row+dof_yy));


                    // y=0 face
                    //sum the contribution from elem i-1 on the opposite face
                    eval(out(row+dof_x)) += eval(in2(j-1, row+dof_xx));

                    //sum the contribution from elem k-1 on the opposite face
                    eval(out(row+dof_z)) += eval(in2(j-1, row+dof_zz));


                    // x=0 face
                    //sum the contribution from elem j-1 on the opposite face
                    eval(out(row+dof_y)) += eval(in2(i-1, row+dof_yy));

                    //sum the contribution from elem k-1 on the opposite face
                    eval(out(row+dof_z)) += eval(in2(i-1, row+dof_zz));

                }

            //edges: setting nodes internal to the edges
            for(short_t I=1; I<N1-1; I++)
            {
                auto dof_x=indexing.index(I, 0, 0);
                auto dof_xx=indexing.index(I, N2, N3);
                auto dof_xy=indexing.index(I, N2, 0);
                auto dof_xz=indexing.index(I, 0, N3);

                auto dof_y=indexing.index(0, I, 0);
                auto dof_yy=indexing.index(N1, I, N3);
                auto dof_yx=indexing.index(N1, I, 0);
                auto dof_yz=indexing.index(0, I, N3);

                auto dof_z=indexing.index(0, 0, I);
                auto dof_zz=indexing.index(N1, N2, I);
                auto dof_zy=indexing.index(0, N2, I);
                auto dof_zx=indexing.index(N1, 0, I);

                eval(out(row+dof_x)) += eval(in2(j-1,k-1, row+dof_xx))
                    + eval(in2(k-1, row+dof_xz))
                    + eval(in2(j-1, row+dof_xy));

                eval(out(row+dof_y)) += eval(in2(i-1,k-1, row+dof_yy))
                    + eval(in2(i-1, row+dof_yx))
                    + eval(in2(k-1, row+dof_yz));

                eval(out(row+dof_z)) += eval(in2(j-1,i-1, row+dof_zz))
                    + eval(in2(j-1, row+dof_zy))
                    + eval(in2(i-1, row+dof_zx));

            }

            //corner cases, setting (0,0,0)
            auto dof_111=indexing.index(N1, N2, N3);
            auto dof_110=indexing.index(N1, N2, 0);
            auto dof_101=indexing.index(N1, 0, N3);
            auto dof_100=indexing.index(N1, 0, 0);
            auto dof_011=indexing.index(0, N2, N3);
            auto dof_010=indexing.index(0, N1, 0);
            auto dof_001=indexing.index(0,0,N3);
            auto dof_000=indexing.index(0,0,0);

            eval(out(// row+dof_000
                     )) += eval(in2(i-1,j-1,k-1, row+dof_111))
                + eval(in2(i-1,j-1, row+dof_110))
                + eval(in2(i-1,k-1, row+dof_101))
                + eval(in2(k-1,j-1, row+dof_011))
                + eval(in2(k-1, row+dof_001))
                + eval(in2(j-1, row+dof_010))
                + eval(in2(i-1, row+dof_100));

            // //sum the contribution from elem k-1 on the opposite face
            // eval(out(row+dof_z)) += Operator()(eval(in1(row+dof_z)), eval(in2(i-1, row+dof_zz)));

        }

    };



    // [uniform]
    /**
       @class functor copying the repeated values on the matching dofs

       used only for post-processing purposes
     */
    template<typename Geometry>
    struct uniform {

        using geo_map=typename Geometry::geo_map;

        using in1=gt::accessor<0, enumtype::in, gt::extent<> , 4>;
        using out=gt::accessor<1, enumtype::inout, gt::extent<> , 4> ;
        using arg_list=boost::mpl::vector<in1, out> ;

        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_interval) {
            gt::dimension<1> i;
            gt::dimension<2> j;
            gt::dimension<3> k;
            gt::dimension<4> row;


            //hypothesis here: the cardinaxlity is order^3 (isotropic 3D tensor product element)
#ifdef __CUDACC__
#ifdef NDEBUG
            constexpr
#endif
                gt::meta_storage_base<static_int<__COUNTER__>,gt::layout_map<2,1,0>,false> indexing{static_int<3>(), static_int<3>(), static_int<3>()};
#else
#ifdef NDEBUG
            constexpr
#endif
                gt::meta_storage_base<static_int<__COUNTER__>,gt::layout_map<2,1,0>,false> indexing{Geometry::geo_map::order+1, Geometry::geo_map::order+1, Geometry::geo_map::order+1};

#endif
            uint_t N1 = indexing.template dim<0>()-1;
            uint_t N2 = indexing.template dim<1>()-1;
            uint_t N3 = indexing.template dim<2>()-1;

            //for all dofs in a boundary face (supposing that the dofs per face are the same)
            for(short_t I=1; I<indexing.template dim<0>()-1; I++)
                for(short_t J=1; J<indexing.template dim<1>()-1; J++)
                {
                    //for each (3) faces
                    auto dof_x=indexing.index(0, (int)I, (int)J);
                    auto dof_xx=indexing.index(indexing.template dim<0>()-1, I, J);
                    auto dof_y=indexing.index(I, 0, J);
                    auto dof_yy=indexing.index(I, indexing.template dim<1>()-1, J);
                    auto dof_z=indexing.index(I, J, 0);
                    auto dof_zz=indexing.index(I, J, indexing.template dim<2>()-1);

                    //replace the value from elem i-1 on the opposite face
                    eval(out(i-1, row+dof_xx)) = eval(in1(row+dof_x));

                    //replace the value from elem j-1 on the opposite face
                    eval(out(j-1, row+dof_yy)) = eval(in1(row+dof_y));

                    //replace the value from elem k-1 on the opposite face
                    eval(out(k-1, row+dof_zz)) = eval(in1(row+dof_z));
                }

            //corner cases, setting (0,0,0)
            auto dof_111=indexing.index(N1, N2, N3);
            auto dof_110=indexing.index(N1, N2, 0);
            auto dof_101=indexing.index(N1, 0, N3);
            auto dof_100=indexing.index(N1, 0, 0);
            auto dof_011=indexing.index(0, N2, N3);
            auto dof_010=indexing.index(0, N1, 0);
            auto dof_001=indexing.index(0,0,N3);
            auto dof_000=indexing.index(0,0,0);

            eval(out()) = eval(in1());
            eval(out(i-1,j-1,k-1, row+dof_111)) = eval(in1(row+dof_000));
            eval(out(i-1,j-1, row+dof_110)) = eval(in1(row+dof_000));
            eval(out(i-1,k-1, row+dof_101)) = eval(in1(row+dof_000));
            eval(out(k-1,j-1, row+dof_011)) = eval(in1(row+dof_000));
            eval(out(k-1, row+dof_001)) = eval(in1(row+dof_000));
            eval(out(j-1, row+dof_010)) = eval(in1(row+dof_000));
            eval(out(i-1, row+dof_100)) = eval(in1(row+dof_000));

        }
    };

    // TODO: debug only functor, to be deleted
    // TODO: this is a low performance temporary global assembly functor
    // Stencil points correspond to global dof pairs (P,Q)
    struct global_assemble {

        using in=gt::accessor<0, enumtype::in, gt::extent<0,0,0,0> , 5> ;
        using in_map=gt::accessor<1, enumtype::in, gt::extent<0,0,0> , 4>;
        using out=gt::accessor<2, enumtype::inout, gt::extent<0,0,0,0> , 5> ;
        using arg_list=boost::mpl::vector<in, in_map, out> ;

        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_interval) {

        	// Retrieve elements dof grid gt::dimensions and number of dofs per element
            const uint_t d1=eval.get().template get_storage_dim<0>(in());
            const uint_t d2=eval.get().template get_storage_dim<1>(in());
            const uint_t d3=eval.get().template get_storage_dim<2>(in());
            const uint_t basis_cardinality=eval.get().template get_storage_dim<3>(in());

            // Retrieve global dof pair of current stencil point
            // TODO: the computation is positional by default only in debug mode!
            const u_int my_P = eval.i();
            const u_int my_Q = eval.j()%eval.get().template get_storage_dim<0>(out());

            // Loop over element dofs
            for(u_int i=0;i<d1;++i)
            {
                for(u_int j=0;j<d2;++j)
                {
                    for(u_int k=0;k<d3;++k)
                    {
                    // Loop over single element dofs
                        for(u_short l_dof1=0;l_dof1<basis_cardinality;++l_dof1)
                        {
                            const u_int P=eval(!in_map(i,j,k,l_dof1));

                            if(P == my_P)
                            {
                                for(u_short l_dof2=0;l_dof2<basis_cardinality;++l_dof2)
                                {
                                    const u_int Q=eval(!in_map(i,j,k,l_dof2));

                                    if(Q == my_Q)
                                    {
                                    // Current local dof pair corresponds to global dof
                                    // stencil point, update global matrix
                                        eval(out(0,0,0,0,0)) += eval(!in(i,j,k,l_dof1,l_dof2));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    };

    // TODO: debug only functor, to be deleted
    // TODO: this is an updated version of the "global_assemble" functor without ifs and conditional branchings
    struct global_assemble_no_if {

          using in=gt::accessor<0, enumtype::in, gt::extent<0,0,0,0> , 5> ;
          using in_map=gt::accessor<1, enumtype::in, gt::extent<0,0,0> , 4>;
          using out=gt::accessor<2, enumtype::inout, gt::extent<0,0,0,0> , 3> ;
          using arg_list=boost::mpl::vector<in, in_map, out> ;

          template <typename Evaluation>
          GT_FUNCTION
          static void Do(Evaluation const & eval, x_interval) {

              // Retrieve elements dof grid gt::dimensions and number of dofs per element
              const uint_t d1=eval.get().template get_storage_dim<0>(in());
              const uint_t d2=eval.get().template get_storage_dim<1>(in());
              const uint_t d3=eval.get().template get_storage_dim<2>(in());
              const uint_t basis_cardinality=eval.get().template get_storage_dim<3>(in());

              // Retrieve global dof pair of current stencil point
              // TODO: the computation is positional by default only in debug mode!
              const u_int my_P = eval.i();
              const u_int my_Q = eval.j()%eval.get().template get_storage_dim<0>(out());

              // Loop over element dofs
              for(u_int i=0;i<d1;++i)
              {
                  for(u_int j=0;j<d2;++j)
                  {
                      for(u_int k=0;k<d3;++k)
                      {
                          // Loop over single element dofs
                          for(u_short l_dof1=0;l_dof1<basis_cardinality;++l_dof1)
                          {
                              // TODO: check next line
                              const u_int P_fact(eval(!in_map(i,j,k,l_dof1))==my_P);
                              for(u_short l_dof2=0;l_dof2<basis_cardinality;++l_dof2)
                              {
                                  // Current local dof pair corresponds to global dof
                                  // stencil point, update global matrix
                                  eval(out(0,0,0)) += (eval(!in_map(i,j,k,l_dof2))==my_Q)*P_fact*eval(!in(i,j,k,l_dof1,l_dof2));
                              }
                          }
                      }
                  }
              }
          }
      };

    // TODO: debug only functor, to be deleted
    // TODO: vector (single dof indexed object) assemble functor
    struct global_vector_assemble_no_if {

        using in=gt::accessor<0, enumtype::in, gt::extent<0,0,0,0> , 4> ;
        using in_map=gt::accessor<1, enumtype::in, gt::extent<0,0,0> , 4>;
        using out=gt::accessor<2, enumtype::inout, gt::extent<0,0,0,0> , 3> ;
        using arg_list=boost::mpl::vector<in, in_map, out> ;

        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_interval) {

            // Retrieve elements dof grid gt::dimensions and number of dofs per element
            const uint_t d1=eval.get().template get_storage_dim<0>(in());
            const uint_t d2=eval.get().template get_storage_dim<1>(in());
            const uint_t d3=eval.get().template get_storage_dim<2>(in());
            const uint_t basis_cardinality=eval.get().template get_storage_dim<3>(in());

            // Retrieve global dof pair of current stencil point
            // TODO: the computation is positional by default only in debug mode!
            const u_int my_P = eval.i();

            // Loop over element dofs
            for(u_int i=0;i<d1;++i)
            {
                for(u_int j=0;j<d2;++j)
                {
                    for(u_int k=0;k<d3;++k)
                    {
                        // Loop over single element dofs
                        for(u_short l_dof1=0;l_dof1<basis_cardinality;++l_dof1)
                        {
                            // Current local dof corresponds to global dof
                            // stencil point, update global vector
                            eval(!out(my_P,0,0)) += (eval(!in_map(i,j,k,l_dof1))==my_P)*eval(!in(i,j,k,l_dof1));
                        }
                    }
                }
            }
        }
    };

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


    // TODO: doxy
    template <ushort_t N_DOF0, ushort_t N_DOF1, ushort_t N_DOF2>
    struct hexahedron_matrix_distribute {

        using inout=gt::accessor<0, enumtype::inout, gt::extent<> , 5>;
        using arg_list=boost::mpl::vector<inout> ;

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

                            eval(inout(i-1,dof1+indexing.index(I1,J1,indexing.template dim<2>()-1),dof2+indexing.index(I2,J2,indexing.template dim<2>()-1))) =
                                eval(inout(dof1+indexing.index(I1,J1,0),dof2+indexing.index(I2,J2,0)));

                            eval(inout(j-1,dof1+indexing.index(J1,indexing.template dim<1>()-1,I1),dof2+indexing.index(J2,indexing.template dim<1>()-1,I2))) =
                                eval(inout(dof1+indexing.index(J1,0,I1),dof2+indexing.index(J2,0,I2)));

                            eval(inout(k-1,dof1+indexing.index(indexing.template dim<0>()-1,I1,J1),dof2+indexing.index(indexing.template dim<0>()-1,I2,J2))) =
                                eval(inout(dof1+indexing.index(0,I1,J1),dof2+indexing.index(0,I2,J2)));


                        }

            // 2 (A,F+B+G)+(F+B+G,A)
            short_t J2=0;
            for(short_t I1=1; I1<indexing.template dim<0>()-1; I1++)
                for(short_t J1=1; J1<indexing.template dim<1>()-1; J1++)
                    for(short_t I2=0; I2<indexing.template dim<0>(); I2++)
                    {
                        eval(inout(i-1,dof1+indexing.index(I1,J1,indexing.template dim<2>()-1),dof2+indexing.index(I2,J2,indexing.template dim<2>()-1))) =
                            eval(inout(dof1+indexing.index(I1,J1,0),dof2+indexing.index(I2,J2,0)));
                        eval(inout(i-1,dof1+indexing.index(I2,J2,indexing.template dim<2>()-1),dof2+indexing.index(I1,J1,indexing.template dim<2>()-1))) =
                            eval(inout(dof1+indexing.index(I2,J2,0),dof2+indexing.index(I1,J1,0)));

                        eval(inout(j-1,dof1+indexing.index(J1,indexing.template dim<1>()-1,I1),dof2+indexing.index(J2,indexing.template dim<1>()-1,I2))) =
                            eval(inout(dof1+indexing.index(J1,0,I1),dof2+indexing.index(J2,0,I2)));
                        eval(inout(j-1,dof1+indexing.index(J2,indexing.template dim<1>()-1,I2),dof2+indexing.index(J1,indexing.template dim<1>()-1,I1))) =
                            eval(inout(dof1+indexing.index(J2,0,I2),dof2+indexing.index(J1,0,I1)));

                        eval(inout(k-1,dof1+indexing.index(indexing.template dim<0>()-1,I1,J1),dof2+indexing.index(indexing.template dim<0>()-1,I2,J2))) =
                            eval(inout(dof1+indexing.index(0,I1,J1),dof2+indexing.index(0,I2,J2)));
                        eval(inout(k-1,dof1+indexing.index(indexing.template dim<0>()-1,I2,J2),dof2+indexing.index(indexing.template dim<0>()-1,I1,J1))) =
                            eval(inout(dof1+indexing.index(0,I2,J2),dof2+indexing.index(0,I1,J1)));

                    }

            // 3 (A,H+D+I)+(H+D+I,A)
            J2=indexing.template dim<1>()-1;
            for(short_t I1=1; I1<indexing.template dim<0>()-1; I1++)
                for(short_t J1=1; J1<indexing.template dim<1>()-1; J1++)
                    for(short_t I2=0; I2<indexing.template dim<0>(); I2++)
                    {
                        eval(inout(i-1,dof1+indexing.index(I1,J1,indexing.template dim<2>()-1),dof2+indexing.index(I2,J2,indexing.template dim<2>()-1))) =
                            eval(inout(dof1+indexing.index(I1,J1,0),dof2+indexing.index(I2,J2,0)));
                        eval(inout(i-1,dof1+indexing.index(I2,J2,indexing.template dim<2>()-1),dof2+indexing.index(I1,J1,indexing.template dim<2>()-1))) =
                            eval(inout(dof1+indexing.index(I2,J2,0),dof2+indexing.index(I1,J1,0)));

                        eval(inout(j-1,dof1+indexing.index(J1,indexing.template dim<1>()-1,I1),dof2+indexing.index(J2,indexing.template dim<1>()-1,I2))) =
                            eval(inout(dof1+indexing.index(J1,0,I1),dof2+indexing.index(J2,0,I2)));
                        eval(inout(j-1,dof1+indexing.index(J2,indexing.template dim<1>()-1,I2),dof2+indexing.index(J1,indexing.template dim<1>()-1,I1))) =
                            eval(inout(dof1+indexing.index(J2,0,I2),dof2+indexing.index(J1,0,I1)));


                        eval(inout(k-1,dof1+indexing.index(indexing.template dim<0>()-1,I1,J1),dof2+indexing.index(indexing.template dim<0>()-1,I2,J2))) =
                            eval(inout(dof1+indexing.index(0,I1,J1),dof2+indexing.index(0,I2,J2)));
                        eval(inout(k-1,dof1+indexing.index(indexing.template dim<0>()-1,I2,J2),dof2+indexing.index(indexing.template dim<0>()-1,I1,J1))) =
                            eval(inout(dof1+indexing.index(0,I2,J2),dof2+indexing.index(0,I1,J1)));

                    }

            // 4 (C,B+A+D+G+E+I)+(B+A+D+G+E+I,C)
            short_t I1=0;
            for(short_t J1=1; J1<indexing.template dim<1>()-1; J1++)
                for(short_t I2=1; I2<indexing.template dim<0>(); I2++)
                    for(short_t J2=0; J2<indexing.template dim<1>(); J2++)
                    {

                        eval(inout(i-1,dof1+indexing.index(I1,J1,indexing.template dim<2>()-1),dof2+indexing.index(I2,J2,indexing.template dim<2>()-1))) =
                            eval(inout(dof1+indexing.index(I1,J1,0),dof2+indexing.index(I2,J2,0)));
                        eval(inout(i-1,dof1+indexing.index(I2,J2,indexing.template dim<2>()-1),dof2+indexing.index(I1,J1,indexing.template dim<2>()-1))) =
                           eval(inout(dof1+indexing.index(I2,J2,0),dof2+indexing.index(I1,J1,0)));

                        eval(inout(j-1,dof1+indexing.index(J1,indexing.template dim<1>()-1,I1),dof2+indexing.index(J2,indexing.template dim<1>()-1,I2))) =
                           eval(inout(dof1+indexing.index(J1,0,I1),dof2+indexing.index(J2,0,I2)));
                        eval(inout(j-1,dof1+indexing.index(J2,indexing.template dim<1>()-1,I2),dof2+indexing.index(J1,indexing.template dim<1>()-1,I1))) =
                           eval(inout(dof1+indexing.index(J2,0,I2),dof2+indexing.index(J1,0,I1)));

                        eval(inout(k-1,dof1+indexing.index(indexing.template dim<0>()-1,I1,J1),dof2+indexing.index(indexing.template dim<0>()-1,I2,J2))) =
                           eval(inout(dof1+indexing.index(0,I1,J1),dof2+indexing.index(0,I2,J2)));
                        eval(inout(k-1,dof1+indexing.index(indexing.template dim<0>()-1,I2,J2),dof2+indexing.index(indexing.template dim<0>()-1,I1,J1))) =
                           eval(inout(dof1+indexing.index(0,I2,J2),dof2+indexing.index(0,I1,J1)));

                    }

            // 5 (B,H+D+I)+(H+D+I,B)
            short_t J1=0;
            J2=indexing.template dim<1>() - 1;
            for(short_t I1=1; I1<indexing.template dim<0>()-1; I1++)
                for(short_t I2=0; I2<indexing.template dim<0>(); I2++)
                {

                    eval(inout(i-1,dof1+indexing.index(I1,J1,indexing.template dim<2>()-1),dof2+indexing.index(I2,J2,indexing.template dim<2>()-1))) =
                        eval(inout(dof1+indexing.index(I1,J1,0),dof2+indexing.index(I2,J2,0)));
                    eval(inout(i-1,dof1+indexing.index(I2,J2,indexing.template dim<2>()-1),dof2+indexing.index(I1,J1,indexing.template dim<2>()-1))) =
                        eval(inout(dof1+indexing.index(I2,J2,0),dof2+indexing.index(I1,J1,0)));

                    eval(inout(j-1,dof1+indexing.index(J1,indexing.template dim<1>()-1,I1),dof2+indexing.index(J2,indexing.template dim<1>()-1,I2))) =
                        eval(inout(dof1+indexing.index(J1,0,I1),dof2+indexing.index(J2,0,I2)));
                    eval(inout(j-1,dof1+indexing.index(J2,indexing.template dim<1>()-1,I2),dof2+indexing.index(J1,indexing.template dim<1>()-1,I1))) =
                        eval(inout(dof1+indexing.index(J2,0,I2),dof2+indexing.index(J1,0,I1)));

                    eval(inout(k-1,dof1+indexing.index(indexing.template dim<0>()-1,I1,J1),dof2+indexing.index(indexing.template dim<0>()-1,I2,J2))) =
                        eval(inout(dof1+indexing.index(0,I1,J1),dof2+indexing.index(0,I2,J2)));
                    eval(inout(k-1,dof1+indexing.index(indexing.template dim<0>()-1,I2,J2),dof2+indexing.index(indexing.template dim<0>()-1,I1,J1))) =
                        eval(inout(dof1+indexing.index(0,I2,J2),dof2+indexing.index(0,I1,J1)));

                }

            // 6 (E,B+A+D)+(B+A+D,E)
            I1=indexing.template dim<0>()-1;
            for(short_t J1=1; J1<indexing.template dim<1>()-1; J1++)
                for(short_t I2=1; I2<indexing.template dim<0>()-1; I2++)
                    for(short_t J2=0; J2<indexing.template dim<1>(); J2++)
                    {

                        eval(inout(i-1,dof1+indexing.index(I1,J1,indexing.template dim<2>()-1),dof2+indexing.index(I2,J2,indexing.template dim<2>()-1))) =
                            eval(inout(dof1+indexing.index(I1,J1,0),dof2+indexing.index(I2,J2,0)));
                        eval(inout(i-1,dof1+indexing.index(I2,J2,indexing.template dim<2>()-1),dof2+indexing.index(I1,J1,indexing.template dim<2>()-1))) =
                            eval(inout(dof1+indexing.index(I2,J2,0),dof2+indexing.index(I1,J1,0)));

                        eval(inout(j-1,dof1+indexing.index(J1,indexing.template dim<1>()-1,I1),dof2+indexing.index(J2,indexing.template dim<1>()-1,I2))) =
                            eval(inout(dof1+indexing.index(J1,0,I1),dof2+indexing.index(J2,0,I2)));
                        eval(inout(j-1,dof1+indexing.index(J2,indexing.template dim<1>()-1,I2),dof2+indexing.index(J1,indexing.template dim<1>()-1,I1))) =
                            eval(inout(dof1+indexing.index(J2,0,I2),dof2+indexing.index(J1,0,I1)));

                        eval(inout(k-1,dof1+indexing.index(indexing.template dim<0>()-1,I1,J1),dof2+indexing.index(indexing.template dim<0>()-1,I2,J2))) =
                            eval(inout(dof1+indexing.index(0,I1,J1),dof2+indexing.index(0,I2,J2)));
                        eval(inout(k-1,dof1+indexing.index(indexing.template dim<0>()-1,I2,J2),dof2+indexing.index(indexing.template dim<0>()-1,I1,J1))) =
                            eval(inout(dof1+indexing.index(0,I2,J2),dof2+indexing.index(0,I1,J1)));
                    }



            // 7 (F,D+I)+(D+I,F)
            I1=0;
            J1=0;
            J2=indexing.template dim<1>() - 1;
            for(short_t I2=1; I2<indexing.template dim<0>(); I2++)
            {

                eval(inout(i-1,dof1+indexing.index(I1,J1,indexing.template dim<2>()-1),dof2+indexing.index(I2,J2,indexing.template dim<2>()-1))) =
                    eval(inout(dof1+indexing.index(I1,J1,0),dof2+indexing.index(I2,J2,0)));
                eval(inout(i-1,dof1+indexing.index(I2,J2,indexing.template dim<2>()-1),dof2+indexing.index(I1,J1,indexing.template dim<2>()-1))) =
                    eval(inout(dof1+indexing.index(I2,J2,0),dof2+indexing.index(I1,J1,0)));

                eval(inout(j-1,dof1+indexing.index(J1,indexing.template dim<1>()-1,I1),dof2+indexing.index(J2,indexing.template dim<1>()-1,I2))) =
                    eval(inout(dof1+indexing.index(J1,0,I1),dof2+indexing.index(J2,0,I2)));
                eval(inout(j-1,dof1+indexing.index(J2,indexing.template dim<1>()-1,I2),dof2+indexing.index(J1,indexing.template dim<1>()-1,I1))) =
                    eval(inout(dof1+indexing.index(J2,0,I2),dof2+indexing.index(J1,0,I1)));

                eval(inout(k-1,dof1+indexing.index(indexing.template dim<0>()-1,I1,J1),dof2+indexing.index(indexing.template dim<0>()-1,I2,J2))) =
                    eval(inout(dof1+indexing.index(0,I1,J1),dof2+indexing.index(0,I2,J2)));
                eval(inout(k-1,dof1+indexing.index(indexing.template dim<0>()-1,I2,J2),dof2+indexing.index(indexing.template dim<0>()-1,I1,J1))) =
                    eval(inout(dof1+indexing.index(0,I2,J2),dof2+indexing.index(0,I1,J1)));

            }

            // 8 (G,H+D)+(H+D,G)
            I1=indexing.template dim<1>() - 1;
            J1=0;
            J2=indexing.template dim<1>() - 1;
            for(short_t I2=0; I2<indexing.template dim<0>()-1; I2++)
            {

                eval(inout(i-1,dof1+indexing.index(I1,J1,indexing.template dim<2>()-1),dof2+indexing.index(I2,J2,indexing.template dim<2>()-1))) =
                    eval(inout(dof1+indexing.index(I1,J1,0),dof2+indexing.index(I2,J2,0)));
                eval(inout(i-1,dof1+indexing.index(I2,J2,indexing.template dim<2>()-1),dof2+indexing.index(I1,J1,indexing.template dim<2>()-1))) =
                    eval(inout(dof1+indexing.index(I2,J2,0),dof2+indexing.index(I1,J1,0)));

                eval(inout(j-1,dof1+indexing.index(J1,indexing.template dim<1>()-1,I1),dof2+indexing.index(J2,indexing.template dim<1>()-1,I2))) =
                    eval(inout(dof1+indexing.index(J1,0,I1),dof2+indexing.index(J2,0,I2)));
                eval(inout(j-1,dof1+indexing.index(J2,indexing.template dim<1>()-1,I2),dof2+indexing.index(J1,indexing.template dim<1>()-1,I1))) =
                    eval(inout(dof1+indexing.index(J2,0,I2),dof2+indexing.index(J1,0,I1)));

                eval(inout(k-1,dof1+indexing.index(indexing.template dim<0>()-1,I1,J1),dof2+indexing.index(indexing.template dim<0>()-1,I2,J2))) =
                    eval(inout(dof1+indexing.index(0,I1,J1),dof2+indexing.index(0,I2,J2)));
                eval(inout(k-1,dof1+indexing.index(indexing.template dim<0>()-1,I2,J2),dof2+indexing.index(indexing.template dim<0>()-1,I1,J1))) =
                    eval(inout(dof1+indexing.index(0,I2,J2),dof2+indexing.index(0,I1,J1)));

            }

            // 9 (F,E)+(E,F)
            I1=0;
            J1=0;
            short_t I2=indexing.template dim<0>()-1;
            for(short_t J2=1; J2<indexing.template dim<1>()-1; J2++)
            {

                eval(inout(i-1,dof1+indexing.index(I1,J1,indexing.template dim<2>()-1),dof2+indexing.index(I2,J2,indexing.template dim<2>()-1))) =
                    eval(inout(dof1+indexing.index(I1,J1,0),dof2+indexing.index(I2,J2,0)));
                eval(inout(i-1,dof1+indexing.index(I2,J2,indexing.template dim<2>()-1),dof2+indexing.index(I1,J1,indexing.template dim<2>()-1))) =
                    eval(inout(dof1+indexing.index(I2,J2,0),dof2+indexing.index(I1,J1,0)));

                eval(inout(j-1,dof1+indexing.index(J1,indexing.template dim<1>()-1,I1),dof2+indexing.index(J2,indexing.template dim<1>()-1,I2))) =
                    eval(inout(dof1+indexing.index(J1,0,I1),dof2+indexing.index(J2,0,I2)));
                eval(inout(j-1,dof1+indexing.index(J2,indexing.template dim<1>()-1,I2),dof2+indexing.index(J1,indexing.template dim<1>()-1,I1))) =
                    eval(inout(dof1+indexing.index(J2,0,I2),dof2+indexing.index(J1,0,I1)));

                eval(inout(k-1,dof1+indexing.index(indexing.template dim<0>()-1,I1,J1),dof2+indexing.index(indexing.template dim<0>()-1,I2,J2))) =
                    eval(inout(dof1+indexing.index(0,I1,J1),dof2+indexing.index(0,I2,J2)));
                eval(inout(k-1,dof1+indexing.index(indexing.template dim<0>()-1,I2,J2),dof2+indexing.index(indexing.template dim<0>()-1,I1,J1))) =
                    eval(inout(dof1+indexing.index(0,I2,J2),dof2+indexing.index(0,I1,J1)));

            }

            // 10 (H,E)+(E,H)
            I1=0;
            J1=indexing.template dim<1>()-1;
            I2=indexing.template dim<0>()-1;
            for(short_t J2=1; J2<indexing.template dim<1>()-1; J2++)
            {

                eval(inout(i-1,dof1+indexing.index(I1,J1,indexing.template dim<2>()-1),dof2+indexing.index(I2,J2,indexing.template dim<2>()-1))) =
                    eval(inout(dof1+indexing.index(I1,J1,0),dof2+indexing.index(I2,J2,0)));
                eval(inout(i-1,dof1+indexing.index(I2,J2,indexing.template dim<2>()-1),dof2+indexing.index(I1,J1,indexing.template dim<2>()-1))) =
                    eval(inout(dof1+indexing.index(I2,J2,0),dof2+indexing.index(I1,J1,0)));

                eval(inout(j-1,dof1+indexing.index(J1,indexing.template dim<1>()-1,I1),dof2+indexing.index(J2,indexing.template dim<1>()-1,I2))) =
                    eval(inout(dof1+indexing.index(J1,0,I1),dof2+indexing.index(J2,0,I2)));
                eval(inout(j-1,dof1+indexing.index(J2,indexing.template dim<1>()-1,I2),dof2+indexing.index(J1,indexing.template dim<1>()-1,I1))) =
                    eval(inout(dof1+indexing.index(J2,0,I2),dof2+indexing.index(J1,0,I1)));

                eval(inout(k-1,dof1+indexing.index(indexing.template dim<0>()-1,I1,J1),dof2+indexing.index(indexing.template dim<0>()-1,I2,J2))) =
                    eval(inout(dof1+indexing.index(0,I1,J1),dof2+indexing.index(0,I2,J2)));
                eval(inout(k-1,dof1+indexing.index(indexing.template dim<0>()-1,I2,J2),dof2+indexing.index(indexing.template dim<0>()-1,I1,J1))) =
                    eval(inout(dof1+indexing.index(0,I2,J2),dof2+indexing.index(0,I1,J1)));

            }

            // 11 (F,B+G)+(B+G,F)
            I1=0;
            J1=0;
            J2=0;
            for(short_t I2=1; I2<indexing.template dim<0>(); I2++)
            {

                eval(inout(i-1,dof1+indexing.index(I1,J1,indexing.template dim<2>()-1),dof2+indexing.index(I2,J2,indexing.template dim<2>()-1))) =
                    eval(inout(dof1+indexing.index(I1,J1,0),dof2+indexing.index(I2,J2,0)));
                eval(inout(j-1,dof1+indexing.index(I1,indexing.template dim<1>()-1,0),dof2+indexing.index(I2,indexing.template dim<1>()-1,0))) =
                    eval(inout(dof1+indexing.index(I1,J1,0),dof2+indexing.index(I2,J2,0)));
                eval(inout(i-1,j-1,dof1+indexing.index(I1,indexing.template dim<1>()-1,indexing.template dim<2>()-1),dof2+indexing.index(I2,indexing.template dim<1>()-1,indexing.template dim<2>()-1))) =
                    eval(inout(dof1+indexing.index(I1,J1,0),dof2+indexing.index(I2,J2,0)));

                eval(inout(i-1,dof1+indexing.index(I2,J2,indexing.template dim<2>()-1),dof2+indexing.index(I1,J1,indexing.template dim<2>()-1)))  =
                    eval(inout(dof1+indexing.index(I2,J2,0),dof2+indexing.index(I1,J1,0)));
                eval(inout(j-1,dof1+indexing.index(I2,indexing.template dim<1>()-1,0),dof2+indexing.index(I1,indexing.template dim<1>()-1,0)))  =
                    eval(inout(dof1+indexing.index(I2,J2,0),dof2+indexing.index(I1,J1,0)));
                eval(inout(i-1,j-1,dof1+indexing.index(I2,indexing.template dim<1>()-1,indexing.template dim<2>()-1),dof2+indexing.index(I1,indexing.template dim<1>()-1,indexing.template dim<2>()-1))) =
                    eval(inout(dof1+indexing.index(I2,J2,0),dof2+indexing.index(I1,J1,0)));

                eval(inout(j-1,dof1+indexing.index(J1,indexing.template dim<1>()-1,I1),dof2+indexing.index(J2,indexing.template dim<1>()-1,I2)))  =
                    eval(inout(dof1+indexing.index(J1,0,I1),dof2+indexing.index(J2,0,I2)));
                eval(inout(k-1,dof1+indexing.index(indexing.template dim<0>()-1,0,I1),dof2+indexing.index(indexing.template dim<0>()-1,0,I2)))  =
                    eval(inout(dof1+indexing.index(J1,0,I1),dof2+indexing.index(J2,0,I2)));
                eval(inout(j-1,k-1,dof1+indexing.index(indexing.template dim<0>()-1,indexing.template dim<1>()-1,I1),dof2+indexing.index(indexing.template dim<0>()-1,indexing.template dim<1>()-1,I2))) =
                    eval(inout(dof1+indexing.index(J1,0,I1),dof2+indexing.index(J2,0,I2)));

                eval(inout(j-1,dof1+indexing.index(J2,indexing.template dim<1>()-1,I2),dof2+indexing.index(J1,indexing.template dim<1>()-1,I1)))  =
                    eval(inout(dof1+indexing.index(J2,0,I2),dof2+indexing.index(J1,0,I1)));
                eval(inout(k-1,dof1+indexing.index(indexing.template dim<0>()-1,0,I2),dof2+indexing.index(indexing.template dim<0>()-1,0,I1)))  =
                    eval(inout(dof1+indexing.index(J2,0,I2),dof2+indexing.index(J1,0,I1)));
                eval(inout(j-1,k-1,dof1+indexing.index(indexing.template dim<0>()-1,indexing.template dim<1>()-1,I2),dof2+indexing.index(indexing.template dim<0>()-1,indexing.template dim<1>()-1,I1))) =
                    eval(inout(dof1+indexing.index(J2,0,I2),dof2+indexing.index(J1,0,I1)));

                eval(inout(k-1,dof1+indexing.index(indexing.template dim<0>()-1,I1,J1),dof2+indexing.index(indexing.template dim<0>()-1,I2,J2)))  =
                    eval(inout(dof1+indexing.index(0,I1,J1),dof2+indexing.index(0,I2,J2)));
                eval(inout(i-1,dof1+indexing.index(0,I1,indexing.template dim<2>()-1),dof2+indexing.index(0,I2,indexing.template dim<2>()-1)))  =
                    eval(inout(dof1+indexing.index(0,I1,J1),dof2+indexing.index(0,I2,J2)));
                eval(inout(i-1,k-1,dof1+indexing.index(indexing.template dim<0>()-1,I1,indexing.template dim<2>()-1),dof2+indexing.index(indexing.template dim<0>()-1,I2,indexing.template dim<2>()-1))) =
                    eval(inout(dof1+indexing.index(0,I1,J1),dof2+indexing.index(0,I2,J2)));

                eval(inout(k-1,dof1+indexing.index(indexing.template dim<0>()-1,I2,J2),dof2+indexing.index(indexing.template dim<0>()-1,I1,J1)))  =
                    eval(inout(dof1+indexing.index(0,I2,J2),dof2+indexing.index(0,I1,J1)));
                eval(inout(i-1,dof1+indexing.index(0,I2,indexing.template dim<2>()-1),dof2+indexing.index(0,I1,indexing.template dim<2>()-1)))  =
                    eval(inout(dof1+indexing.index(0,I2,J2),dof2+indexing.index(0,I1,J1)));
                eval(inout(i-1,k-1,dof1+indexing.index(indexing.template dim<0>()-1,I2,indexing.template dim<2>()-1),dof2+indexing.index(indexing.template dim<0>()-1,I1,indexing.template dim<2>()-1))) =
                    eval(inout(dof1+indexing.index(0,I2,J2),dof2+indexing.index(0,I1,J1)));


            }

            // 12 (G,B)+(B,G)
            I1=indexing.template dim<0>()-1;
            J1=0;
            J2=0;
            for(short_t I2=1; I2<indexing.template dim<0>()-1; I2++)
            {

                eval(inout(i-1,dof1+indexing.index(I1,J1,indexing.template dim<2>()-1),dof2+indexing.index(I2,J2,indexing.template dim<2>()-1))) =
                    eval(inout(dof1+indexing.index(I1,J1,0),dof2+indexing.index(I2,J2,0)));
                eval(inout(j-1,dof1+indexing.index(I1,indexing.template dim<1>()-1,0),dof2+indexing.index(I2,indexing.template dim<1>()-1,0))) =
                    eval(inout(dof1+indexing.index(I1,J1,0),dof2+indexing.index(I2,J2,0)));
                eval(inout(i-1,j-1,dof1+indexing.index(I1,indexing.template dim<1>()-1,indexing.template dim<2>()-1),dof2+indexing.index(I2,indexing.template dim<1>()-1,indexing.template dim<2>()-1))) =
                    eval(inout(dof1+indexing.index(I1,J1,0),dof2+indexing.index(I2,J2,0)));

                eval(inout(i-1,dof1+indexing.index(I2,J2,indexing.template dim<2>()-1),dof2+indexing.index(I1,J1,indexing.template dim<2>()-1))) =
                    eval(inout(dof1+indexing.index(I2,J2,0),dof2+indexing.index(I1,J1,0)));
                eval(inout(j-1,dof1+indexing.index(I2,indexing.template dim<1>()-1,0),dof2+indexing.index(I1,indexing.template dim<1>()-1,0))) =
                    eval(inout(dof1+indexing.index(I2,J2,0),dof2+indexing.index(I1,J1,0)));
                eval(inout(i-1,j-1,dof1+indexing.index(I2,indexing.template dim<1>()-1,indexing.template dim<2>()-1),dof2+indexing.index(I1,indexing.template dim<1>()-1,indexing.template dim<2>()-1))) =
                    eval(inout(dof1+indexing.index(I2,J2,0),dof2+indexing.index(I1,J1,0)));

                eval(inout(j-1,dof1+indexing.index(J1,indexing.template dim<1>()-1,I1),dof2+indexing.index(J2,indexing.template dim<1>()-1,I2))) =
                    eval(inout(dof1+indexing.index(J1,0,I1),dof2+indexing.index(J2,0,I2)));
                eval(inout(k-1,dof1+indexing.index(indexing.template dim<0>()-1,0,I1),dof2+indexing.index(indexing.template dim<0>()-1,0,I2))) =
                    eval(inout(dof1+indexing.index(J1,0,I1),dof2+indexing.index(J2,0,I2)));
                eval(inout(j-1,k-1,dof1+indexing.index(indexing.template dim<0>()-1,indexing.template dim<1>()-1,I1),dof2+indexing.index(indexing.template dim<0>()-1,indexing.template dim<1>()-1,I2))) =
                    eval(inout(dof1+indexing.index(J1,0,I1),dof2+indexing.index(J2,0,I2)));

                eval(inout(j-1,dof1+indexing.index(J2,indexing.template dim<1>()-1,I2),dof2+indexing.index(J1,indexing.template dim<1>()-1,I1))) =
                    eval(inout(dof1+indexing.index(J2,0,I2),dof2+indexing.index(J1,0,I1)));
                eval(inout(k-1,dof1+indexing.index(indexing.template dim<0>()-1,0,I2),dof2+indexing.index(indexing.template dim<0>()-1,0,I1))) =
                    eval(inout(dof1+indexing.index(J2,0,I2),dof2+indexing.index(J1,0,I1)));
                eval(inout(j-1,k-1,dof1+indexing.index(indexing.template dim<0>()-1,indexing.template dim<1>()-1,I2),dof2+indexing.index(indexing.template dim<0>()-1,indexing.template dim<1>()-1,I1))) =
                    eval(inout(dof1+indexing.index(J2,0,I2),dof2+indexing.index(J1,0,I1)));

                eval(inout(k-1,dof1+indexing.index(indexing.template dim<0>()-1,I1,J1),dof2+indexing.index(indexing.template dim<0>()-1,I2,J2))) =
                    eval(inout(dof1+indexing.index(0,I1,J1),dof2+indexing.index(0,I2,J2)));
                eval(inout(i-1,dof1+indexing.index(0,I1,indexing.template dim<2>()-1),dof2+indexing.index(0,I2,indexing.template dim<2>()-1))) =
                    eval(inout(dof1+indexing.index(0,I1,J1),dof2+indexing.index(0,I2,J2)));
                eval(inout(i-1,k-1,dof1+indexing.index(indexing.template dim<0>()-1,I1,indexing.template dim<2>()-1),dof2+indexing.index(indexing.template dim<0>()-1,I2,indexing.template dim<2>()-1))) =
                    eval(inout(dof1+indexing.index(0,I1,J1),dof2+indexing.index(0,I2,J2)));

                eval(inout(k-1,dof1+indexing.index(indexing.template dim<0>()-1,I2,J2),dof2+indexing.index(indexing.template dim<0>()-1,I1,J1))) =
                    eval(inout(dof1+indexing.index(0,I2,J2),dof2+indexing.index(0,I1,J1)));
                eval(inout(i-1,dof1+indexing.index(0,I2,indexing.template dim<2>()-1),dof2+indexing.index(0,I1,indexing.template dim<2>()-1))) =
                    eval(inout(dof1+indexing.index(0,I2,J2),dof2+indexing.index(0,I1,J1)));
                eval(inout(i-1,k-1,dof1+indexing.index(indexing.template dim<0>()-1,I2,indexing.template dim<2>()-1),dof2+indexing.index(indexing.template dim<0>()-1,I1,indexing.template dim<2>()-1))) =
                    eval(inout(dof1+indexing.index(0,I2,J2),dof2+indexing.index(0,I1,J1)));

            }

            // 13 (B,B)
            J1=0;
            J2=0;
            for(short_t I1=1; I1<indexing.template dim<0>()-1; I1++)
                for(short_t I2=1; I2<indexing.template dim<0>()-1; I2++)
                {

                    eval(inout(i-1,dof1+indexing.index(I1,J1,indexing.template dim<2>()-1),dof2+indexing.index(I2,J2,indexing.template dim<2>()-1))) =
                        eval(inout(dof1+indexing.index(I1,J1,0),dof2+indexing.index(I2,J2,0)));
                    eval(inout(j-1,dof1+indexing.index(I1,indexing.template dim<1>()-1,0),dof2+indexing.index(I2,indexing.template dim<1>()-1,0))) =
                        eval(inout(dof1+indexing.index(I1,J1,0),dof2+indexing.index(I2,J2,0)));
                    eval(inout(i-1,j-1,dof1+indexing.index(I1,indexing.template dim<1>()-1,indexing.template dim<2>()-1),dof2+indexing.index(I2,indexing.template dim<1>()-1,indexing.template dim<2>()-1))) =
                        eval(inout(dof1+indexing.index(I1,J1,0),dof2+indexing.index(I2,J2,0)));

                    eval(inout(j-1,dof1+indexing.index(J1,indexing.template dim<1>()-1,I1),dof2+indexing.index(J2,indexing.template dim<1>()-1,I2))) =
                        eval(inout(dof1+indexing.index(J1,0,I1),dof2+indexing.index(J2,0,I2)));
                    eval(inout(k-1,dof1+indexing.index(indexing.template dim<0>()-1,0,I1),dof2+indexing.index(indexing.template dim<0>()-1,0,I2))) =
                        eval(inout(dof1+indexing.index(J1,0,I1),dof2+indexing.index(J2,0,I2)));
                    eval(inout(j-1,k-1,dof1+indexing.index(indexing.template dim<0>()-1,indexing.template dim<1>()-1,I1),dof2+indexing.index(indexing.template dim<0>()-1,indexing.template dim<1>()-1,I2))) =
                        eval(inout(dof1+indexing.index(J1,0,I1),dof2+indexing.index(J2,0,I2)));

                    eval(inout(k-1,dof1+indexing.index(indexing.template dim<0>()-1,I1,J1),dof2+indexing.index(indexing.template dim<0>()-1,I2,J2))) =
                        eval(inout(dof1+indexing.index(0,I1,J1),dof2+indexing.index(0,I2,J2)));
                    eval(inout(i-1,dof1+indexing.index(0,I1,indexing.template dim<2>()-1),dof2+indexing.index(0,I2,indexing.template dim<2>()-1))) =
                        eval(inout(dof1+indexing.index(0,I1,J1),dof2+indexing.index(0,I2,J2)));
                    eval(inout(i-1,k-1,dof1+indexing.index(indexing.template dim<0>()-1,I1,indexing.template dim<2>()-1),dof2+indexing.index(indexing.template dim<0>()-1,I2,indexing.template dim<2>()-1))) =
                        eval(inout(dof1+indexing.index(0,I1,J1),dof2+indexing.index(0,I2,J2)));

                }

            // 14 (F,F)
            eval(inout(i-1,dof1+indexing.index(0,0,indexing.template dim<2>()-1),dof2+indexing.index(0,0,indexing.template dim<2>()-1))) =
                eval(inout(dof1+0,dof2+0));
            eval(inout(j-1,dof1+indexing.index(0,indexing.template dim<1>()-1,0),dof2+indexing.index(0,indexing.template dim<1>()-1,0))) =
                eval(inout(dof1+0,dof2+0));
            eval(inout(i-1,j-1,dof1+indexing.index(0,indexing.template dim<1>()-1,indexing.template dim<2>()-1),dof2+indexing.index(0,indexing.template dim<1>()-1,indexing.template dim<2>()-1))) =
                eval(inout(dof1+0,dof2+0));
            eval(inout(k-1,dof1+indexing.index(indexing.template dim<0>()-1,0,0),dof2+indexing.index(indexing.template dim<0>()-1,0,0))) =
                eval(inout(dof1+0,dof2+0));
            eval(inout(i-1,k-1,dof1+indexing.index(indexing.template dim<0>()-1,0,indexing.template dim<2>()-1),dof2+indexing.index(indexing.template dim<0>()-1,0,indexing.template dim<2>()-1))) =
                eval(inout(dof1+0,dof2+0));
            eval(inout(j-1,k-1,dof1+indexing.index(indexing.template dim<0>()-1,indexing.template dim<1>()-1,0),dof2+indexing.index(indexing.template dim<0>()-1,indexing.template dim<1>()-1,0))) =
                eval(inout(dof1+0,dof2+0));
            eval(inout(i-1,j-1,k-1,
                       dof1+indexing.index(indexing.template dim<0>()-1,indexing.template dim<1>()-1,indexing.template dim<2>()-1),
                       dof2+indexing.index(indexing.template dim<0>()-1,indexing.template dim<1>()-1,indexing.template dim<2>()-1))) =
                eval(inout(dof1+0,dof2+0));

        }
    };



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

//    template <ushort_t N_DOF0, ushort_t N_DOF1, ushort_t N_DOF2>
//    struct hexahedron_assemble_shrinking{
//
//        // TODO: this functor can be factorized with the corresponding one for assembly computation
//        // TODO: look at hexahedron_assemble for strategy explanation
//        // TODO: matrix symmetry hypothesis not required here (but we know that matrix for this functor are symmetric)
//        // TODO: extend to non isotropic dof distribution case
//        // TODO: expression within loops are always the same, avoid code duplication
//        using in=gt::accessor<0, enumtype::in, gt::extent<> , 5>;
//        using out=gt::accessor<1, enumtype::inout, gt::extent<> , 5> ;
//        using arg_list=boost::mpl::vector<in, out> ;
//
//        template <typename Evaluation>
//        GT_FUNCTION
//        static void Do(Evaluation const & eval, x_interval) {
//
//
//            gt::dimension<4> dof1;
//            gt::dimension<5> dof2;
//
//            constexpr gt::meta_storage_base<__COUNTER__,gt::layout_map<2,1,0>,false> indexing{N_DOF0,N_DOF1,N_DOF2};
//
//            uint_t dof1_index(0);
//            uint_t dof2_index(0);
//
//            // 0 (INTERNAL,INTERNAL)
//            for(short_t I1=1; I1<indexing.template dim<0>()-1; I1++)
//                for(short_t J1=1; J1<indexing.template dim<1>()-1; J1++)
//                    for(short_t K1=1; K1<indexing.template dim<2>()-1; K1++,dof1_index++)
//                        for(short_t I2=1; I2<indexing.template dim<0>()-1; I2++)
//                            for(short_t J2=1; J2<indexing.template dim<1>()-1; J2++)
//                                for(short_t K2=1; K2<indexing.template dim<2>()-1; K2++,dof2_index++)
//                                {
//                                    eval(out(dof1+dof1_index,dof2+dof2_index)) =
//                                            eval(in(dof1+indexing.index(I1,J1,K1),dof2+indexing.index(I2,J2,K2)));
//                                }
//
//            MISSING INTERNAL,SURFACE
//
//            // TODO: performance: it should be better to split loop for the single faces in order to have sequential memory access
//            uint_t face_stride_11 = (indexing.template dim<0>()-2)*(indexing.template dim<1>()-2);
//            uint_t face_stride_12 = (indexing.template dim<0>()-2)*(indexing.template dim<1>()-2);
//            uint_t face_stride_21 = 2*face_stride_11;
//            uint_t face_stride_22 = 2*face_stride_12;
//            // 1 (A,A)
//            for(short_t I1=1; I1<indexing.template dim<0>()-1; I1++)
//                for(short_t J1=1; J1<indexing.template dim<1>()-1; J1++,dof1_index++)
//                    for(short_t I2=1; I2<indexing.template dim<0>()-1; I2++)
//                        for(short_t J2=1; J2<indexing.template dim<1>()-1; J2++,dof2_index++)
//                        {
//                            eval(out(dof1+dof1_index,dof2+dof2_index)) =
//                                    eval(in(dof1+indexing.index(I1,J1,0),dof2+indexing.index(I2,J2,0)));
//
//                            eval(out(dof1+(dof1_index+face_stride_11),dof2+(dof2_index+face_stride_12))) =
//                                    eval(in(dof1+indexing.index(J1,0,I1),dof2+indexing.index(J2,0,I2)));
//
//                            eval(out(dof1+(dof1_index+face_stride_21),dof2+(dof2_index+face_stride_22))) =
//                                    eval(in(dof1+indexing.index(0,I1,J1),dof2+indexing.index(0,I2,J2)));
//                        }
//
//            // TODO: fix dofx_index update when loop over JX is missing!
//            // 2 (A,F+B+G)+(F+B+G,A)
//            face_stride_11 = (indexing.template dim<0>()-2)*(indexing.template dim<1>()-2);
//            face_stride_12 = face_stride_11;
//            face_stride_21 = 2*face_stride_11;
//            face_stride_22 = 2*face_stride_12;
//            short_t J2=0;
//            for(short_t I1=1; I1<indexing.template dim<0>()-1; I1++)
//                for(short_t J1=1; J1<indexing.template dim<1>()-1; J1++,dof1_index++){
//
//                     dof2_index = dof1_index;
//
//                    for(short_t I2=0; I2<indexing.template dim<0>(); I2++,dof2_index++)
//                    {
//                        eval(out(dof1+dof1_index,dof2+dof2_index)) =
//                                eval(in(dof1+indexing.index(I1,J1,0),dof2+indexing.index(I2,J2,0)));
//
//                        eval(out(dof1+dof2_index,dof2+dof1_index)) =
//                                eval(in(dof1+indexing.index(I2,J2,0),dof2+indexing.index(I1,J1,0)));
//
//
//                        eval(out(dof1+(dof1_index+face_stride_11),dof2+(dof2_index+face_stride_12))) =
//                                eval(out(dof1+indexing.index(J1,0,I1),dof2+indexing.index(J2,0,I2)));
//
//                        eval(out(dof1+(dof2_index+face_stride_12),dof2+(dof1_index+face_stride_11))) =
//                                eval(out(dof1+indexing.index(J2,0,I2),dof2+indexing.index(J1,0,I1)));
//
//
//                        eval(out(dof1+(dof1_index+face_stride_21),dof2+(dof2_index+face_stride_22))) =
//                                eval(out(dof1+indexing.index(0,I1,J1),dof2+indexing.index(0,I2,J2)));
//
//                        eval(out(dof1+(dof2_index+face_stride_22),dof2+(dof1_index+face_stride_21))) =
//                                eval(out(dof1+indexing.index(0,I2,J2),dof2+indexing.index(0,I1,J1)));
//
//                    }
//                }
//
//            // TODO: fix dofx_index update when loop over JX is missing!
//            // 3 (A,H+D+I)+(H+D+I,A)
//            face_stride_11 = (indexing.template dim<0>()-2)*(indexing.template dim<1>()-2);
//            face_stride_12 = face_stride_11;
//            face_stride_21 = 2*face_stride_11;
//            face_stride_22 = 2*face_stride_12;
//            J2=indexing.template dim<1>()-1;
//            for(short_t I1=1; I1<indexing.template dim<0>()-1; I1++)
//                for(short_t J1=1; J1<indexing.template dim<1>()-1; J1++,dof1_index++)
//                {
//                    dof2_index = dof1_index;
//
//                    for(short_t I2=0; I2<indexing.template dim<0>(); I2++,dof2_index++)
//                    {
//
//                            eval(out(dof1+dof1_index,dof2+dof2_index)) =
//                                    eval(in(dof1+indexing.index(I1,J1,0),dof2+indexing.index(I2,J2,0)));
//
//                            eval(out(dof1+dof2_index,dof2+dof1_index)) =
//                                    eval(in(dof1+indexing.index(I2,J2,0),dof2+indexing.index(I1,J1,0)));
//
//
//                            eval(out(dof1+(dof1_index+face_stride_11),dof2+(dof2_index+face_stride_12))) =
//                                    eval(out(dof1+indexing.index(J1,0,I1),dof2+indexing.index(J2,0,I2)));
//
//                            eval(out(dof1+(dof2_index+face_stride_12),dof2+(dof1_index+face_stride_11))) =
//                                    eval(out(dof1+indexing.index(J2,0,I2),dof2+indexing.index(J1,0,I1)));
//
//
//                            eval(out(dof1+(dof1_index+face_stride_21),dof2+(dof2_index+face_stride_22))) =
//                                    eval(out(dof1+indexing.index(0,I1,J1),dof2+indexing.index(0,I2,J2)));
//
//                            eval(out(dof1+(dof2_index+face_stride_22),dof2+(dof1_index+face_stride_21))) =
//                                    eval(out(dof1+indexing.index(0,I2,J2),dof2+indexing.index(0,I1,J1)));
//
//                    }
//                }
//
//            // 4 (C,B+A+D+G+E+I)+(B+A+D+G+E+I,C)
//            face_stride_11 = (indexing.template dim<1>()-2);
//            face_stride_12 = face_stride_11;
//            face_stride_21 = 2*face_stride_11;
//            face_stride_22 = 2*face_stride_12;
//            short_t I1=0;
//            for(short_t J1=1; J1<indexing.template dim<1>()-1; J1++,dof1_index++)
//            {
//                dof2_index = dof1_index;
//                for(short_t I2=1; I2<indexing.template dim<0>(); I2++)
//                    for(short_t J2=0; J2<indexing.template dim<1>(); J2++,dof2_index++)
//                    {
//
//                            eval(out(dof1+dof1_index,dof2+dof2_index)) =
//                                    eval(in(dof1+indexing.index(I1,J1,0),dof2+indexing.index(I2,J2,0)));
//
//                            eval(out(dof1+dof2_index,dof2+dof1_index)) =
//                                    eval(in(dof1+indexing.index(I2,J2,0),dof2+indexing.index(I1,J1,0)));
//
//
//                            eval(out(dof1+(dof1_index+face_stride_11),dof2+(dof2_index+face_stride_12))) =
//                                    eval(out(dof1+indexing.index(J1,0,I1),dof2+indexing.index(J2,0,I2)));
//
//                            eval(out(dof1+(dof2_index+face_stride_12),dof2+(dof1_index+face_stride_11))) =
//                                    eval(out(dof1+indexing.index(J2,0,I2),dof2+indexing.index(J1,0,I1)));
//
//
//                            eval(out(dof1+(dof1_index+face_stride_21),dof2+(dof2_index+face_stride_22))) =
//                                    eval(out(dof1+indexing.index(0,I1,J1),dof2+indexing.index(0,I2,J2)));
//
//                            eval(out(dof1+(dof2_index+face_stride_22),dof2+(dof1_index+face_stride_21))) =
//                                    eval(out(dof1+indexing.index(0,I2,J2),dof2+indexing.index(0,I1,J1)));
//
//                    }
//            }
//
//            // TODO: fix dofx_index update when loop over JX is missing!
//            // 5 (B,H+D+I)+(H+D+I,B)
//            face_stride_11 = (indexing.template dim<1>()-2);
//            face_stride_12 = face_stride_11;
//            face_stride_21 = 2*face_stride_11;
//            face_stride_22 = 2*face_stride_12;
//            short_t J1=0;
//            J2=indexing.template dim<1>() - 1;
//            for(short_t I1=1; I1<indexing.template dim<0>()-1; I1++,dof1_index++)
//            {
//                dof2_index = dof1_index;
//
//                for(short_t I2=0; I2<indexing.template dim<0>(); I2++,dof2_index++)
//                {
//
//                        eval(out(dof1+dof1_index,dof2+dof2_index)) =
//                                eval(in(dof1+indexing.index(I1,J1,0),dof2+indexing.index(I2,J2,0)));
//
//                        eval(out(dof1+dof2_index,dof2+dof1_index)) =
//                                eval(in(dof1+indexing.index(I2,J2,0),dof2+indexing.index(I1,J1,0)));
//
//
//                        eval(out(dof1+(dof1_index+face_stride_11),dof2+(dof2_index+face_stride_12))) =
//                                eval(out(dof1+indexing.index(J1,0,I1),dof2+indexing.index(J2,0,I2)));
//
//                        eval(out(dof1+(dof2_index+face_stride_12),dof2+(dof1_index+face_stride_11))) =
//                                eval(out(dof1+indexing.index(J2,0,I2),dof2+indexing.index(J1,0,I1)));
//
//
//                        eval(out(dof1+(dof1_index+face_stride_21),dof2+(dof2_index+face_stride_22))) =
//                                eval(out(dof1+indexing.index(0,I1,J1),dof2+indexing.index(0,I2,J2)));
//
//                        eval(out(dof1+(dof2_index+face_stride_22),dof2+(dof1_index+face_stride_21))) =
//                                eval(out(dof1+indexing.index(0,I2,J2),dof2+indexing.index(0,I1,J1)));
//
//                }
//            }
//
//            // TODO: fix dofx_index update when loop over JX is missing!
//            // 6 (E,B+A+D)+(B+A+D,E)
//            face_stride_11 = (indexing.template dim<1>()-2);
//            face_stride_12 = face_stride_11;
//            face_stride_21 = 2*face_stride_11;
//            face_stride_22 = 2*face_stride_12;
//            I1=indexing.template dim<0>()-1;
//            for(short_t J1=1; J1<indexing.template dim<1>()-1; J1++,dof1_index++)
//            {
//                dof2_index = dof1_index;
//                for(short_t I2=1; I2<indexing.template dim<0>()-1; I2++)
//                    for(short_t J2=0; J2<indexing.template dim<1>(); J2++,dof2_index++)
//                    {
//                            eval(out(dof1+dof1_index,dof2+dof2_index)) =
//                                    eval(in(dof1+indexing.index(I1,J1,0),dof2+indexing.index(I2,J2,0)));
//
//                            eval(out(dof1+dof2_index,dof2+dof1_index)) =
//                                    eval(in(dof1+indexing.index(I2,J2,0),dof2+indexing.index(I1,J1,0)));
//
//
//                            eval(out(dof1+(dof1_index+face_stride_11),dof2+(dof2_index+face_stride_12))) =
//                                    eval(out(dof1+indexing.index(J1,0,I1),dof2+indexing.index(J2,0,I2)));
//
//                            eval(out(dof1+(dof2_index+face_stride_12),dof2+(dof1_index+face_stride_11))) =
//                                    eval(out(dof1+indexing.index(J2,0,I2),dof2+indexing.index(J1,0,I1)));
//
//
//                            eval(out(dof1+(dof1_index+face_stride_21),dof2+(dof2_index+face_stride_22))) =
//                                    eval(out(dof1+indexing.index(0,I1,J1),dof2+indexing.index(0,I2,J2)));
//
//                            eval(out(dof1+(dof2_index+face_stride_22),dof2+(dof1_index+face_stride_21))) =
//                                    eval(out(dof1+indexing.index(0,I2,J2),dof2+indexing.index(0,I1,J1)));
//                    }
//            }
//
//
//            // TODO: fix dofx_index update when loop over JX is missing!
//            // 7 (F,D+I)+(D+I,F)
//            face_stride_11 = 1;
//            face_stride_12 = face_stride_11;
//            face_stride_21 = 2*face_stride_11;
//            face_stride_22 = 2*face_stride_12;
//            dof2_index = dof1_index;
//            I1=0;
//            J1=0;
//            J2=indexing.template dim<1>() - 1;
//            for(short_t I2=1; I2<indexing.template dim<0>(); I2++,dof2_index++)
//            {
//
//                    eval(out(dof1+dof1_index,dof2+dof2_index)) =
//                            eval(in(dof1+indexing.index(I1,J1,0),dof2+indexing.index(I2,J2,0)));
//
//                    eval(out(dof1+dof2_index,dof2+dof1_index)) =
//                            eval(in(dof1+indexing.index(I2,J2,0),dof2+indexing.index(I1,J1,0)));
//
//
//                    eval(out(dof1+(dof1_index+face_stride_11),dof2+(dof2_index+face_stride_12))) =
//                            eval(out(dof1+indexing.index(J1,0,I1),dof2+indexing.index(J2,0,I2)));
//
//                    eval(out(dof1+(dof2_index+face_stride_12),dof2+(dof1_index+face_stride_11))) =
//                            eval(out(dof1+indexing.index(J2,0,I2),dof2+indexing.index(J1,0,I1)));
//
//
//                    eval(out(dof1+(dof1_index+face_stride_21),dof2+(dof2_index+face_stride_22))) =
//                            eval(out(dof1+indexing.index(0,I1,J1),dof2+indexing.index(0,I2,J2)));
//
//                    eval(out(dof1+(dof2_index+face_stride_22),dof2+(dof1_index+face_stride_21))) =
//                            eval(out(dof1+indexing.index(0,I2,J2),dof2+indexing.index(0,I1,J1)));
//
//            }
//            dof1_index++;
//
//            // TODO: fix dofx_index update when loop over JX is missing!
//            // 8 (G,H+D)+(H+D,G)
//            face_stride_11 = 1;
//            face_stride_12 = face_stride_11;
//            face_stride_21 = 2*face_stride_11;
//            face_stride_22 = 2*face_stride_12;
//            dof2_index = dof1_index;
//            I1=indexing.template dim<1>() - 1;
//            J1=0;
//            J2=indexing.template dim<1>() - 1;
//            for(short_t I2=0; I2<indexing.template dim<0>()-1; I2++,dof2_index++)
//            {
//
//                    eval(out(dof1+dof1_index,dof2+dof2_index)) =
//                            eval(in(dof1+indexing.index(I1,J1,0),dof2+indexing.index(I2,J2,0)));
//
//                    eval(out(dof1+dof2_index,dof2+dof1_index)) =
//                            eval(in(dof1+indexing.index(I2,J2,0),dof2+indexing.index(I1,J1,0)));
//
//
//                    eval(out(dof1+(dof1_index+face_stride_11),dof2+(dof2_index+face_stride_12))) =
//                            eval(out(dof1+indexing.index(J1,0,I1),dof2+indexing.index(J2,0,I2)));
//
//                    eval(out(dof1+(dof2_index+face_stride_12),dof2+(dof1_index+face_stride_11))) =
//                            eval(out(dof1+indexing.index(J2,0,I2),dof2+indexing.index(J1,0,I1)));
//
//
//                    eval(out(dof1+(dof1_index+face_stride_21),dof2+(dof2_index+face_stride_22))) =
//                            eval(out(dof1+indexing.index(0,I1,J1),dof2+indexing.index(0,I2,J2)));
//
//                    eval(out(dof1+(dof2_index+face_stride_22),dof2+(dof1_index+face_stride_21))) =
//                            eval(out(dof1+indexing.index(0,I2,J2),dof2+indexing.index(0,I1,J1)));
//
//            }
//            dof1_index++;
//
//            // TODO: fix dofx_index update when loop over JX is missing!
//            // 9 (F,E)+(E,F)
//            face_stride_11 = 1;
//            face_stride_12 = face_stride_11;
//            face_stride_21 = 2*face_stride_11;
//            face_stride_22 = 2*face_stride_12;
//            dof2_index = dof1_index;
//            I1=0;
//            J1=0;
//            short_t I2=indexing.template dim<0>()-1;
//            for(short_t J2=1; J2<indexing.template dim<1>()-1; J2++,dof2_index++)
//            {
//
//                    eval(out(dof1+dof1_index,dof2+dof2_index)) =
//                            eval(in(dof1+indexing.index(I1,J1,0),dof2+indexing.index(I2,J2,0)));
//
//                    eval(out(dof1+dof2_index,dof2+dof1_index)) =
//                            eval(in(dof1+indexing.index(I2,J2,0),dof2+indexing.index(I1,J1,0)));
//
//
//                    eval(out(dof1+(dof1_index+face_stride_11),dof2+(dof2_index+face_stride_12))) =
//                            eval(out(dof1+indexing.index(J1,0,I1),dof2+indexing.index(J2,0,I2)));
//
//                    eval(out(dof1+(dof2_index+face_stride_12),dof2+(dof1_index+face_stride_11))) =
//                            eval(out(dof1+indexing.index(J2,0,I2),dof2+indexing.index(J1,0,I1)));
//
//
//                    eval(out(dof1+(dof1_index+face_stride_21),dof2+(dof2_index+face_stride_22))) =
//                            eval(out(dof1+indexing.index(0,I1,J1),dof2+indexing.index(0,I2,J2)));
//
//                    eval(out(dof1+(dof2_index+face_stride_22),dof2+(dof1_index+face_stride_21))) =
//                            eval(out(dof1+indexing.index(0,I2,J2),dof2+indexing.index(0,I1,J1)));
//
//            }
//            dof1_index++;
//
//            // TODO: fix dofx_index update when loop over JX is missing!
//            // 10 (H,E)+(E,H)
//            face_stride_11 = 1;
//            face_stride_12 = face_stride_11;
//            face_stride_21 = 2*face_stride_11;
//            face_stride_22 = 2*face_stride_12;
//            dof2_index = dof1_index;
//            I1=0;
//            J1=indexing.template dim<1>()-1;
//            I2=indexing.template dim<0>()-1;
//            for(short_t J2=1; J2<indexing.template dim<1>()-1; J2++,dof2_index++)
//            {
//
//                    eval(out(dof1+dof1_index,dof2+dof2_index)) =
//                            eval(in(dof1+indexing.index(I1,J1,0),dof2+indexing.index(I2,J2,0)));
//
//                    eval(out(dof1+dof2_index,dof2+dof1_index)) =
//                            eval(in(dof1+indexing.index(I2,J2,0),dof2+indexing.index(I1,J1,0)));
//
//
//                    eval(out(dof1+(dof1_index+face_stride_11),dof2+(dof2_index+face_stride_12))) =
//                            eval(out(dof1+indexing.index(J1,0,I1),dof2+indexing.index(J2,0,I2)));
//
//                    eval(out(dof1+(dof2_index+face_stride_12),dof2+(dof1_index+face_stride_11))) =
//                            eval(out(dof1+indexing.index(J2,0,I2),dof2+indexing.index(J1,0,I1)));
//
//
//                    eval(out(dof1+(dof1_index+face_stride_21),dof2+(dof2_index+face_stride_22))) =
//                            eval(out(dof1+indexing.index(0,I1,J1),dof2+indexing.index(0,I2,J2)));
//
//                    eval(out(dof1+(dof2_index+face_stride_22),dof2+(dof1_index+face_stride_21))) =
//                            eval(out(dof1+indexing.index(0,I2,J2),dof2+indexing.index(0,I1,J1)));
//
//            }
//            dof1_index++;
//
//            // TODO: fix dofx_index update when loop over JX is missing!
//            // 11 (F,B+G)+(B+G,F)
//            face_stride_11 = 1;
//            face_stride_12 = face_stride_11;
//            face_stride_21 = 2*face_stride_11;
//            face_stride_22 = 2*face_stride_12;
//            dof2_index = dof1_index;
//            I1=0;
//            J1=0;
//            J2=0;
//            for(short_t I2=1; I2<indexing.template dim<0>(); I2++,dof2_index++)
//            {
//
//                    eval(out(dof1+dof1_index,dof2+dof2_index)) =
//                            eval(in(dof1+indexing.index(I1,J1,0),dof2+indexing.index(I2,J2,0)));
//
//                    eval(out(dof1+dof2_index,dof2+dof1_index)) =
//                            eval(in(dof1+indexing.index(I2,J2,0),dof2+indexing.index(I1,J1,0)));
//
//
//                    eval(out(dof1+(dof1_index+face_stride_11),dof2+(dof2_index+face_stride_12))) =
//                            eval(out(dof1+indexing.index(J1,0,I1),dof2+indexing.index(J2,0,I2)));
//
//                    eval(out(dof1+(dof2_index+face_stride_12),dof2+(dof1_index+face_stride_11))) =
//                            eval(out(dof1+indexing.index(J2,0,I2),dof2+indexing.index(J1,0,I1)));
//
//
//                    eval(out(dof1+(dof1_index+face_stride_21),dof2+(dof2_index+face_stride_22))) =
//                            eval(out(dof1+indexing.index(0,I1,J1),dof2+indexing.index(0,I2,J2)));
//
//                    eval(out(dof1+(dof2_index+face_stride_22),dof2+(dof1_index+face_stride_21))) =
//                            eval(out(dof1+indexing.index(0,I2,J2),dof2+indexing.index(0,I1,J1)));
//
//            }
//            dof1_index++;
//
//            // TODO: fix dofx_index update when loop over JX is missing!
//            // 12 (G,B)+(B,G)
//            face_stride_11 = 1;
//            face_stride_12 = face_stride_11;
//            face_stride_21 = 2*face_stride_11;
//            face_stride_22 = 2*face_stride_12;
//            dof2_index = dof1_index;
//            I1=indexing.template dim<0>()-1;
//            J1=0;
//            J2=0;
//            for(short_t I2=1; I2<indexing.template dim<0>()-1; I2++,dof2_index++)
//            {
//
//                    eval(out(dof1+dof1_index,dof2+dof2_index)) =
//                            eval(in(dof1+indexing.index(I1,J1,0),dof2+indexing.index(I2,J2,0)));
//
//                    eval(out(dof1+dof2_index,dof2+dof1_index)) =
//                            eval(in(dof1+indexing.index(I2,J2,0),dof2+indexing.index(I1,J1,0)));
//
//
//                    eval(out(dof1+(dof1_index+face_stride_11),dof2+(dof2_index+face_stride_12))) =
//                            eval(out(dof1+indexing.index(J1,0,I1),dof2+indexing.index(J2,0,I2)));
//
//                    eval(out(dof1+(dof2_index+face_stride_12),dof2+(dof1_index+face_stride_11))) =
//                            eval(out(dof1+indexing.index(J2,0,I2),dof2+indexing.index(J1,0,I1)));
//
//
//                    eval(out(dof1+(dof1_index+face_stride_21),dof2+(dof2_index+face_stride_22))) =
//                            eval(out(dof1+indexing.index(0,I1,J1),dof2+indexing.index(0,I2,J2)));
//
//                    eval(out(dof1+(dof2_index+face_stride_22),dof2+(dof1_index+face_stride_21))) =
//                            eval(out(dof1+indexing.index(0,I2,J2),dof2+indexing.index(0,I1,J1)));
//
//            }
//            dof1_index++;
//
//
//            // TODO: fix dofx_index update when loop over JX is missing!
//            // 13 (B,B)
//            face_stride_11 = (indexing.template dim<0>()-2);
//            face_stride_12 = (indexing.template dim<0>()-2);
//            face_stride_21 = 2*face_stride_11;
//            face_stride_22 = 2*face_stride_12;
//            J1=0;
//            J2=0;
//            for(short_t I1=1; I1<indexing.template dim<0>()-1; I1++,dof1_index++)
//                for(short_t I2=1; I2<indexing.template dim<0>()-1; I2++,dof2_index++)
//                {
//
//                        eval(out(dof1+dof1_index,dof2+dof2_index)) =
//                                eval(in(dof1+indexing.index(I1,J1,0),dof2+indexing.index(I2,J2,0)));
//
//                        eval(out(dof1+(dof1_index+face_stride_11),dof2+(dof2_index+face_stride_12))) =
//                                eval(in(dof1+indexing.index(J1,0,I1),dof2+indexing.index(J2,0,I2)));
//
//                        eval(out(dof1+(dof1_index+face_stride_21),dof2+(dof2_index+face_stride_22))) =
//                                eval(in(dof1+indexing.index(0,I1,J1),dof2+indexing.index(0,I2,J2)));
//
//                }
//
//            // 14 (F,F)
//            eval(out(dof1+dof1_index,dof2+dof2_index)) = eval(in(dof1+0,dof2+0));
//
//        }
//
//    };


    // [assemble]

    // TODO: delete and use storage operators instead?
    /**
     * @class Vector copy functor
     */
    template <uint_t N_DOF>
    struct copy_vector {

        using in=gt::accessor<0, enumtype::in, gt::extent<> , 4>;
        using out=gt::accessor<1, enumtype::inout, gt::extent<> , 4> ;
        using arg_list=boost::mpl::vector<in, out> ;

        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_interval) {
            gt::dimension<4> dof;

            for(uint_t i=0; i<N_DOF; i++)
                 eval(out(dof+i)) = eval(in(dof+i));
        }
    };

    /* assigns a field to a constant value**/
    //[assign]
    template< ushort_t Dim, typename Value>
    struct assign;

    template<typename Value>
    struct assign<3,Value> {
        typedef gt::accessor<0, enumtype::inout, gt::extent<0,0,0,0> , 4> field;
        typedef boost::mpl::vector< field > arg_list;

        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_interval) {
                    eval(field())=Value::value;
        }
    };

    template<typename Value>
    struct assign<4,Value> {
        typedef gt::accessor<0, enumtype::inout, gt::extent<0,0,0,0> , 4> field;
        typedef boost::mpl::vector< field > arg_list;

        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_interval) {

            uint_t const num_=eval.get().template get_storage_dim<3>(field());

            for(short_t I=0; I<num_; I++)
                eval(field(gt::dimension<4>(I)))=Value::value;
        }
    };

    template<typename Value>
    struct assign<5,Value> {
        typedef gt::accessor<0, enumtype::inout, gt::extent<0,0,0,0> , 4> field;
        typedef boost::mpl::vector< field > arg_list;

        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_interval) {

            uint_t const dim_1_=eval.get().template get_storage_dim<3>(field());
            uint_t const dim_2_=eval.get().template get_storage_dim<4>(field());

            for(short_t I=0; I<dim_1_; I++)
                for(short_t J=0; J<dim_2_; J++)
                    eval(field(gt::dimension<4>(I), gt::dimension<5>(J)))=Value::value;
        }
    };
    //[assign]

    } // namespace functors
}//namespace gdl
