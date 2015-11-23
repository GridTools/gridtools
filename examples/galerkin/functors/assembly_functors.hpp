#pragma once

namespace functors{

    //watchout: propagating namespace
    using namespace gridtools;
    using namespace expressions;

    typedef gridtools::interval<gridtools::level<0,-1>, gridtools::level<1,-1> > x_interval;
    typedef gridtools::interval<gridtools::level<0,-2>, gridtools::level<1,1> > axis;

    // [update_jac]
    /** updates the values of the Jacobian matrix. The Jacobian matrix, component (i,j) in the quadrature point q, is computed given the geometric map discretization as \f$ J(i,j,q)=\sum_k\frac{\partial \phi_i(x_k,q)}{\partial x_j} x_k \f$
        where x_k are the points in the geometric element*/
    template<typename Geometry, enumtype::Shape S=Geometry::parent_shape >
    struct update_jac{
        using cub=typename Geometry::cub;
        using geo_map=typename Geometry::geo_map;

        typedef accessor<0, range<0,0,0,0> , 5> const grid_points;
        typedef accessor<1, range<0,0,0,0> , 3> const dphi;
        typedef accessor<2, range<0,0,0,0> , 6> jac;
        typedef boost::mpl::vector< grid_points, dphi, jac> arg_list;

        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_interval) {
            dimension<4>::Index qp;
            dimension<5>::Index dimx;
            dimension<6>::Index dimy;
            dimension<1>::Index i;
            dimension<2>::Index j;
            dimension<3>::Index k;

            uint_t const num_cub_points=eval.get().get_storage_dims(dphi())[1];
            uint_t const basis_cardinality=eval.get().get_storage_dims(dphi())[0];

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
                                    eval( jac(dimx+icoor, dimy+jcoor, qp+iter_quad) ) += eval(grid_points(dimension<4>(iterNode), dimension<5>(icoor)) * !dphi(i+iterNode, j+iter_quad, k+jcoor) );
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
    template<typename det>
    struct det_base
	{
        using jac = accessor<0, range<0,0,0,0> , 6> const;
        using jac_det =  accessor<1, range<0,0,0,0> , 4>;
        using arg_list= boost::mpl::vector< jac, jac_det > ;

        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_interval) {
        	det::Do(eval);
        }
	};


    template<typename Geometry, ushort_t Dim=Geometry::geo_map::spaceDim>
    struct det;

    template<typename Geometry>// TODO: number of dimensions can be derived from Geometry
    struct det<Geometry,3> : public det_base< det<Geometry,3> >
    {
        using cub=typename Geometry::cub;
        using super=det_base< det<Geometry,3> >;
        using jacobian= typename super::jac;
        using jacobian_det=typename super::jac_det;

        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_interval) {
            dimension<4>::Index qp;
            dimension<5>::Index dimx;
            dimension<6>::Index dimy;
            uint_t const num_cub_points=eval.get().get_storage_dims(jacobian())[3];

#ifdef __CUDACC__
            assert(num_cub_points==cub::numCubPoints());
#endif
            for(short_t q=0; q< num_cub_points; ++q)
            {
                eval( jacobian_det(qp+q) )= eval(
                    (
		     jacobian(        qp+q)*jacobian(dimx+1, dimy+1, qp+q)*jacobian(dimx+2, dimy+2, qp+q) +
		     jacobian(dimx+1, qp+q)*jacobian(dimx+2, dimy+1, qp+q)*jacobian(dimy+2,         qp+q) +
		     jacobian(dimy+1, qp+q)*jacobian(dimx+1, dimy+2, qp+q)*jacobian(dimx+2,         qp+q) -
		     jacobian(dimy+1, qp+q)*jacobian(dimx+1,         qp+q)*jacobian(dimx+2, dimy+2, qp+q) -
		     jacobian(        qp+q)*jacobian(dimx+2, dimy+1, qp+q)*jacobian(dimx+1, dimy+2, qp+q) -
		     jacobian(dimy+2, qp+q)*jacobian(dimx+1, dimy+1, qp+q)*jacobian(dimx+2,         qp+q)
		     )
	        );
            }
        }
    };


    template<typename Geometry>
    struct det<Geometry,2> : public det_base< det<Geometry,2> >
    {
        using cub=typename Geometry::cub;
        using super=det_base< det<Geometry,2> >;
        using jacobian=typename super::jac;
        using jacobian_det=typename super::jac_det;

        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_interval) {
            dimension<4>::Index qp;
            dimension<5>::Index dimx;
            dimension<6>::Index dimy;
            uint_t const num_cub_points=eval.get().get_storage_dims(jacobian())[3];

#ifdef __CUDACC__
            assert(num_cub_points==cub::numCubPoints());
#endif
            for(short_t q=0; q< num_cub_points; ++q)
            {
                eval( jacobian_det(qp+q) )= eval(
                    (
		     jacobian(        qp+q)*jacobian(dimx+1, dimy+1, qp+q) -
		     jacobian(dimx+1, qp+q)*jacobian(dimy+1, qp+q)
		     )
		    );
            }
        }
    };

    //! [det]

#ifndef __CUDACC__
    //! [inv]
    template <typename Geometry>
    struct inv{
        using cub=typename Geometry::cub;

        //![arguments_inv]
        /**The input arguments to this functors are the matrix and its determinant. */
        using jac      = accessor<0, range<0,0,0,0> , 6> const ;
        using jac_det  = accessor<1, range<0,0,0,0> , 4> const ;
        using jac_inv  = accessor<2, range<0,0,0,0> , 6> ;
        using arg_list = boost::mpl::vector< jac, jac_det, jac_inv>;
        //![arguments_inv]

        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_interval) {
            dimension<4>::Index qp;
            using dimx=dimension<5>;
            using dimy=dimension<6>;
            dimx::Index X;
            dimy::Index Y;
            uint_t const num_cub_points=eval.get().get_storage_dims(jac())[3];

#ifdef __CUDACC__
            assert(num_cub_points==cub::numCubPoints());
#endif

//! [aliases]
            using a_=alias<jac, dimy, dimx>::set<0,0>;
            using b_=alias<jac, dimy, dimx>::set<0,1>;
            using c_=alias<jac, dimy, dimx>::set<0,2>;
            using d_=alias<jac, dimy, dimx>::set<1,0>;
            using e_=alias<jac, dimy, dimx>::set<1,1>;
            using f_=alias<jac, dimy, dimx>::set<1,2>;
            using g_=alias<jac, dimy, dimx>::set<2,0>;
            using h_=alias<jac, dimy, dimx>::set<2,1>;
            using i_=alias<jac, dimy, dimx>::set<2,2>;
//! [aliases]
            // eval( jac(dimx+icoor, dimy+jcoor, qp+iter_quad) )=0.;
            for(short_t q=0; q< num_cub_points; ++q)
            {
                alias<a_, dimension<4> > a(q);
                alias<b_, dimension<4> > b(q);
                alias<c_, dimension<4> > c(q);
                alias<d_, dimension<4> > d(q);
                alias<e_, dimension<4> > e(q);
                alias<f_, dimension<4> > f(q);
                alias<g_, dimension<4> > g(q);
                alias<h_, dimension<4> > h(q);
                alias<i_, dimension<4> > i(q);

                assert(eval(a()) == eval(jac(qp+q)));
                assert(eval(b()) == eval(jac(qp+q, X+1)));
                assert(eval(c()) == eval(jac(qp+q, X+2)));
                assert(eval(d()) == eval(jac(qp+q, Y+1)));

                //! std::cout << "JACOBIAN: "<<std::endl;
                //! std::cout<<eval(a())<<" "<<eval(b())<<" "<<eval(c())<<std::endl;
                //! std::cout<<eval(d())<<" "<<eval(e())<<" "<<eval(f())<<std::endl;
                //! std::cout<<eval(g())<<" "<<eval(h())<<" "<<eval(i())<<std::endl;

                eval( jac_inv(qp+q) )           = eval( ( e()*i() - f()*h())/jac_det(qp+q));
                eval( jac_inv(X+1, qp+q) )      = eval( ( f()*g() - d()*i())/jac_det(qp+q));
                eval( jac_inv(X+2, qp+q) )      = eval( ( d()*h() - e()*g())/jac_det(qp+q));
                eval( jac_inv(Y+1, qp+q) )      = eval( ( c()*h() - b()*i())/jac_det(qp+q));
                eval( jac_inv(Y+1, X+1, qp+q) ) = eval( ( a()*i() - c()*g())/jac_det(qp+q));
                eval( jac_inv(Y+1, X+2, qp+q) ) = eval( ( b()*g() - a()*h())/jac_det(qp+q));
                eval( jac_inv(Y+2, qp+q) )      = eval( ( b()*f() - c()*e())/jac_det(qp+q));
                eval( jac_inv(Y+2, X+1, qp+q) ) = eval( ( c()*d() - a()*f())/jac_det(qp+q));
                eval( jac_inv(Y+2, X+2, qp+q) ) = eval( ( a()*e() - b()*d())/jac_det(qp+q));

                //! std::cout << "JACOBIAN INVERSE: "<<std::endl;
                //! std::cout<<eval(jac_inv(qp+q))<<" "<<eval(jac_inv(qp+q, X+1))<<" "<<eval(jac_inv(qp+q, X+2))<<std::endl;
                //! std::cout<<eval(jac_inv(qp+q, Y+1))<<" "<<eval(jac_inv(qp+q, X+1, Y+1))<<" "<<eval(jac_inv(qp+q, X+2, Y+1))<<std::endl;
                //! std::cout<<eval(jac_inv(qp+q, Y+2))<<" "<<eval(jac_inv(qp+q, X+1, Y+2))<<" "<<eval(jac_inv(qp+q, X+2, Y+2))<<std::endl;
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
    template<typename Geometry, typename Operator>
    struct assemble {

        using geo_map=typename Geometry::geo_map;

        using in1=accessor<0, range<> , 4>;
        using in2=accessor<1, range<> , 4>;
        using out=accessor<2, range<> , 4> ;
        using arg_list=boost::mpl::vector<in1, in2, out> ;

        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_interval) {
            dimension<1>::Index i;
            dimension<2>::Index j;
            dimension<3>::Index k;
            dimension<4>::Index row;


            //hypothesis here: the cardinaxlity is order^3 (isotropic 3D tensor product element)
#ifdef __CUDACC__
            constexpr meta_storage_base<__COUNTER__,layout_map<0,1,2>,false> indexing{static_int<3>(), static_int<3>(), static_int<3>()};
#else
            constexpr meta_storage_base<__COUNTER__,layout_map<0,1,2>,false> indexing{Geometry::geo_map::order+1, Geometry::geo_map::order+1, Geometry::geo_map::order+1};

#endif

            //for all dofs in a boundary face (supposing that the dofs per face are the same)
            for(short_t I=0; I<indexing.template dims<0>(); I++)
                for(short_t J=0; J<indexing.template dims<1>(); J++)
                {

                    //for each (3) faces
                    auto dof_x=indexing.index(0, (int)I, (int)J);
                    auto dof_xx=indexing.index(indexing.template dims<0>()-1, I, J);
                    auto dof_y=indexing.index(I, 0, J);
                    auto dof_yy=indexing.index(I, indexing.template dims<1>()-1, J);
                    auto dof_z=indexing.index(I, J, 0);
                    auto dof_zz=indexing.index(I, J, indexing.template dims<2>()-1);

                    //sum the contribution from elem i-1 on the opposite face
                    eval(out(row+dof_x)) += Operator()(eval(in1(row+dof_x)), eval(in2(i-1, row+dof_xx)));

                    //sum the contribution from elem j-1 on the opposite face
                    eval(out(row+dof_y)) += Operator()(eval(in1(row+dof_y)), eval(in2(j-1, row+dof_yy)));

                    //sum the contribution from elem k-1 on the opposite face
                    eval(out(row+dof_z)) += Operator()(eval(in1(row+dof_z)), eval(in2(k-1, row+dof_zz)));

                }
        }
    };
    // [assemble]


} // namespace functors
