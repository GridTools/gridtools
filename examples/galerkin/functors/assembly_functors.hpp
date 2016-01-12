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

        typedef accessor<0, enumtype::in, extent<0,0,0,0> , 5> const grid_points;
        typedef accessor<1, enumtype::in, extent<0,0,0,0> , 3> const dphi;
        typedef accessor<2, enumtype::inout, extent<0,0,0,0> , 6> jac;
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

            uint_t const num_cub_points=eval.get().template get_storage_dims<1>(dphi());
            uint_t const basis_cardinality=eval.get().template get_storage_dims<0>(dphi());

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
				      eval(grid_points(dimension<4>(iterNode), dimension<5>(icoor)) * !dphi(i+iterNode, j+iter_quad, k+jcoor) );
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
    template<typename Geometry, ushort_t Dim=Geometry::geo_map::spaceDim>
    struct det_impl;

    template<typename Geometry>
    struct det
	{
        using jac = accessor<0, enumtype::in, extent<0,0,0,0> , 6> const;
        using jac_det =  accessor<1, enumtype::inout, extent<0,0,0,0> , 4>;
        using arg_list= boost::mpl::vector< jac, jac_det > ;

        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_interval) {

            dimension<4>::Index qp;
            dimension<5>::Index dimx;
            dimension<6>::Index dimy;
            uint_t const num_cub_points=eval.get().template get_storage_dims<3>(jac());

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
        static void DoCompute(Evaluation const & eval, const dimension<4>::Index& i_qp, const dimension<5>::Index& i_dimx, const dimension<6>::Index& i_dimy, const short_t i_q) {
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
        static void DoCompute(Evaluation const & eval, const dimension<4>::Index& i_qp, const dimension<5>::Index& i_dimx, const dimension<6>::Index& i_dimy, const short_t i_q) {
            eval( jacobian_det(i_qp+i_q) )= eval((
		     jacobian(        i_qp+i_q)*jacobian(i_dimx+1, i_dimy+1, i_qp+i_q) -
		     jacobian(i_dimx+1, i_qp+i_q)*jacobian(i_dimy+1, i_qp+i_q)));
        }

    };
    //! [det]

#ifndef __CUDACC__
    //! [inv]

    template <typename Geometry,ushort_t Dim=Geometry::geo_map::spaceDim>
    struct inv_impl;

    template <typename Geometry>
    struct inv
	{
        using cub=typename Geometry::cub;

        //![arguments_inv]
        /**The input arguments to this functors are the matrix and its determinant. */
        using jac      = accessor<0, enumtype::in, extent<0,0,0,0> , 6> const ;
        using jac_det  = accessor<1, enumtype::in, extent<0,0,0,0> , 4> const ;
        using jac_inv  = accessor<2, enumtype::inout, extent<0,0,0,0> , 6> ;
        using arg_list = boost::mpl::vector< jac, jac_det, jac_inv>;
        //![arguments_inv]

    	template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_interval) {
        	inv_impl<Geometry>::DoCompute(eval);
        }
	};

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

            dimension<4>::Index qp;
            using dimx=dimension<5>;
            using dimy=dimension<6>;
            dimx::Index X;
            dimy::Index Y;
            uint_t const num_cub_points=eval.get().template get_storage_dims<3>(jac());

#ifdef __CUDACC__
            assert(num_cub_points==cub::numCubPoints());
#endif

//! [aliases]
            using a_=typename alias<jac, dimy, dimx>::template set<0,0>;
            using b_=typename alias<jac, dimy, dimx>::template set<0,1>;
            using c_=typename alias<jac, dimy, dimx>::template set<0,2>;
            using d_=typename alias<jac, dimy, dimx>::template set<1,0>;
            using e_=typename alias<jac, dimy, dimx>::template set<1,1>;
            using f_=typename alias<jac, dimy, dimx>::template set<1,2>;
            using g_=typename alias<jac, dimy, dimx>::template set<2,0>;
            using h_=typename alias<jac, dimy, dimx>::template set<2,1>;
            using i_=typename alias<jac, dimy, dimx>::template set<2,2>;
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
            dimension<4>::Index qp;
            using dimx=dimension<5>;
            using dimy=dimension<6>;
            dimx::Index X;
            dimy::Index Y;
            uint_t const num_cub_points=eval.get().template get_storage_dims<3>(jac());

#ifdef __CUDACC__
            assert(num_cub_points==cub::numCubPoints());
#endif

//! [aliases]
            using a_= typename alias<jac, dimy, dimx>::template set<0,0>;
            using b_= typename alias<jac, dimy, dimx>::template set<0,1>;
            using c_= typename alias<jac, dimy, dimx>::template set<1,0>;
            using d_= typename alias<jac, dimy, dimx>::template set<1,1>;
//! [aliases]
            // eval( jac(dimx+icoor, dimy+jcoor, qp+iter_quad) )=0.;
            for(short_t q=0; q< num_cub_points; ++q)
            {
                alias<a_, dimension<4> > a(q);
                alias<b_, dimension<4> > b(q);
                alias<c_, dimension<4> > c(q);
                alias<d_, dimension<4> > d(q);

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

        using in1=accessor<0, enumtype::in, extent<> , 4>;
        using in2=accessor<1, enumtype::in, extent<> , 4>;
        using out=accessor<2, enumtype::inout, extent<> , 4> ;
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
#ifdef NDEBUG
            constexpr
#endif
            meta_storage_base<__COUNTER__,layout_map<0,1,2>,false> indexing{static_int<3>(), static_int<3>(), static_int<3>()};
#else
#ifdef NDEBUG
            constexpr
#endif
                meta_storage_base<__COUNTER__,layout_map<0,1,2>,false> indexing{Geometry::geo_map::order+1, Geometry::geo_map::order+1, Geometry::geo_map::order+1};

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

    // TODO: this is a low performance temporary global assembly functor
    // Stencil points correspond to global dof pairs (P,Q)
    struct global_assemble {

    	using in=accessor<0, enumtype::in, extent<0,0,0,0> , 5> ;
    	using in_map=accessor<1, enumtype::in, extent<0,0,0> , 4>;
    	using out=accessor<2, enumtype::inout, extent<0,0,0,0> , 5> ;
    	using arg_list=boost::mpl::vector<in, in_map, out> ;

        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_interval) {

        	// Retrieve elements dof grid dimensions and number of dofs per element
            const uint_t d1=eval.get().template get_storage_dims<0>(in());
            const uint_t d2=eval.get().template get_storage_dims<1>(in());
            const uint_t d3=eval.get().template get_storage_dims<2>(in());
            const uint_t basis_cardinality=eval.get().template get_storage_dims<3>(in());

            // Retrieve global dof pair of current stencil point
            // TODO: the computation is positional by default only in debug mode!
            const u_int my_P = eval.i();
            const u_int my_Q = eval.j()%eval.get().template get_storage_dims<0>(out());

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

    // TODO: this is an updated version of the "global_assemble" functor without ifs and conditional branchings
    struct global_assemble_no_if {

    	using in=accessor<0, enumtype::in, extent<0,0,0,0> , 5> ;
    	using in_map=accessor<1, enumtype::in, extent<0,0,0> , 4>;
    	using out=accessor<2, enumtype::inout, extent<0,0,0,0> , 5> ;
    	using arg_list=boost::mpl::vector<in, in_map, out> ;

        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_interval) {

        	// Retrieve elements dof grid dimensions and number of dofs per element
            const uint_t d1=eval.get().template get_storage_dims<0>(in());
            const uint_t d2=eval.get().template get_storage_dims<1>(in());
            const uint_t d3=eval.get().template get_storage_dims<2>(in());
            const uint_t basis_cardinality=eval.get().template get_storage_dims<3>(in());

            // Retrieve global dof pair of current stencil point
            // TODO: the computation is positional by default only in debug mode!
            const u_int my_P = eval.i();
            const u_int my_Q = eval.j()%eval.get().template get_storage_dims<0>(out());

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
								eval(out(0,0,0,0,0)) += (eval(!in_map(i,j,k,l_dof2))==my_Q)*P_fact*eval(!in(i,j,k,l_dof1,l_dof2));
							}
						}
					}
				}
			}
		}
    };
    // [assemble]

    /* assigns a field to a constant value**/
    //[zero]
    template< ushort_t Dim, typename T, T Value>
    struct assign;

    template< typename T, T Value>
    struct assign<3, T, Value>{
        typedef accessor<2, enumtype::inout, extent<0,0,0,0> , 3> field;
        typedef boost::mpl::vector< field > arg_list;

        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_interval) {
                    eval(field())=Value;
        }
    };

    template< typename T, T Value>
    struct assign<4,T,Value>{
        typedef accessor<0, enumtype::inout, extent<0,0,0,0> , 4> field;
        typedef boost::mpl::vector< field > arg_list;

        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_interval) {

            uint_t const num_=eval.get().template get_storage_dims<3>(field());

            for(short_t I=0; I<num_; I++)
                eval(field(dimension<4>(I)))=Value;
        }
    };

    template< typename T, T Value>
    struct assign<5,T,Value>{
        typedef accessor<0, enumtype::inout, extent<0,0,0,0> , 5> field;
        typedef boost::mpl::vector< field > arg_list;

        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_interval) {

            uint_t const dim_1_=eval.get().template get_storage_dims<3>(field());
            uint_t const dim_2_=eval.get().template get_storage_dims<4>(field());

            for(short_t I=0; I<dim_1_; I++)
                for(short_t J=0; J<dim_2_; J++)
                    eval(field(dimension<4>(I), dimension<5>(J)))=Value;
        }
    };
    //[zero]

} // namespace functors
