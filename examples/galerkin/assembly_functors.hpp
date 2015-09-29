#pragma once

// namespace gridtools{
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
        typedef accessor<1, range<0,0,0,0> , 6> jac;
        typedef accessor<2, range<0,0,0,0> , 3> const dphi;
        typedef boost::mpl::vector< grid_points, jac, dphi> arg_list;

        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_interval) {
            dimension<4>::Index qp;
            dimension<5>::Index dimx;
            dimension<6>::Index dimy;
            dimension<1>::Index i;
            dimension<2>::Index j;
            dimension<3>::Index k;

            //TODO dimensions should be generic
            for(short_t icoor=0; icoor< shape_property<Geometry::parent_shape>::dimension; ++icoor)
            {
                for(short_t jcoor=0; jcoor< shape_property<S>::dimension; ++jcoor)
                {
                    for(short_t iter_quad=0; iter_quad< cub::numCubPoints()/*quad_pts*/; ++iter_quad)
                    {
                        eval( jac(dimx+icoor, dimy+jcoor, qp+iter_quad) )=0.;
                                for (int_t iterNode=0; iterNode < geo_map::basisCardinality ; ++iterNode)
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
    template<typename Geometry>
    struct det{
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

            for(short_t q=0; q< cub::numCubPoints(); ++q)
            {
                eval( jac_det(qp+q) )= eval(
                    (
                        jac(        qp+q)*jac(dimx+1, dimy+1, qp+q)*jac(dimx+2, dimy+2, qp+q) +
                        jac(dimx+1, qp+q)*jac(dimx+2, dimy+1, qp+q)*jac(dimy+2,         qp+q) +
                        jac(dimy+1, qp+q)*jac(dimx+1, dimy+2, qp+q)*jac(dimx+2,         qp+q) -
                        jac(dimy+1, qp+q)*jac(dimx+1,         qp+q)*jac(dimx+2, dimy+2, qp+q) -
                        jac(        qp+q)*jac(dimx+2, dimy+1, qp+q)*jac(dimx+1, dimy+2, qp+q) -
                        jac(dimy+2, qp+q)*jac(dimx+1, dimy+1, qp+q)*jac(dimx+2,         qp+q)
                        )
                    );
            }
        }
    };
    //! [det]

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
            for(short_t q=0; q< cub::numCubPoints()/*quad_pts*/; ++q)
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

                // std::cout << "JACOBIAN: "<<std::endl;
                // std::cout<<eval(a())<<" "<<eval(b())<<" "<<eval(c())<<std::endl;
                // std::cout<<eval(d())<<" "<<eval(e())<<" "<<eval(f())<<std::endl;
                // std::cout<<eval(g())<<" "<<eval(h())<<" "<<eval(i())<<std::endl;

                eval( jac_inv(qp+q) )           = eval( ( e()*i() - f()*h())/jac_det(qp+q));
                eval( jac_inv(X+1, qp+q) )      = eval( ( f()*g() - d()*i())/jac_det(qp+q));
                eval( jac_inv(X+2, qp+q) )      = eval( ( d()*h() - e()*g())/jac_det(qp+q));
                eval( jac_inv(Y+1, qp+q) )      = eval( ( c()*h() - b()*i())/jac_det(qp+q));
                eval( jac_inv(Y+1, X+1, qp+q) ) = eval( ( a()*i() - c()*g())/jac_det(qp+q));
                eval( jac_inv(Y+1, X+2, qp+q) ) = eval( ( b()*g() - a()*h())/jac_det(qp+q));
                eval( jac_inv(Y+2, qp+q) )      = eval( ( b()*f() - c()*e())/jac_det(qp+q));
                eval( jac_inv(Y+2, X+1, qp+q) ) = eval( ( c()*d() - a()*f())/jac_det(qp+q));
                eval( jac_inv(Y+2, X+2, qp+q) ) = eval( ( a()*e() - b()*d())/jac_det(qp+q));

                // std::cout << "JACOBIAN INVERSE: "<<std::endl;
                // std::cout<<eval(jac_inv(qp+q))<<" "<<eval(jac_inv(qp+q, X+1))<<" "<<eval(jac_inv(qp+q, X+2))<<std::endl;
                // std::cout<<eval(jac_inv(qp+q, Y+1))<<" "<<eval(jac_inv(qp+q, X+1, Y+1))<<" "<<eval(jac_inv(qp+q, X+2, Y+1))<<std::endl;
                // std::cout<<eval(jac_inv(qp+q, Y+2))<<" "<<eval(jac_inv(qp+q, X+1, Y+2))<<" "<<eval(jac_inv(qp+q, X+2, Y+2))<<std::endl;
            }

        }
    };
    // [inv]









    // [assembly]
    template<typename Geometry>
    struct assembly_f {

        using geo_map=typename Geometry::geo_map;

        using in=accessor<0, range<-1,0,-1,0> , 5> const;
        using out=accessor<1, range<0,0,0,0> , 5> ;
        using arg_list=boost::mpl::vector<in, out> ;
        using quad=dimension<4>;

        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_interval) {
            dimension<1>::Index i;
            dimension<2>::Index j;
            dimension<3>::Index k;
            dimension<4>::Index row;
            dimension<5>::Index col;

            // assembly : this part is specific for tensor product topologies
            // points on the edges
            static int_t bd_dim=geo_map::hypercube_t::template boundary_w_dim<1>::n_points::value;

            //for all dofs in a boundary face
            for(short_t I=0; I<bd_dim; I++)
                for(short_t J=0; J<bd_dim; J++)
            //for all dofs in the matching face of a neighbor
            for(short_t II=0; II<bd_dim; II++)
                for(short_t JJ=0; JJ<bd_dim; JJ++)

                {

                    //Hypothesis: the local dofs are ordered according to fe::layout
                    array<int, 3> strides={bd_dim*bd_dim, bd_dim, 1};
                    auto dof_x=(geo_map::layout_t::find<1>(&strides[0]))*I+geo_map::layout_t::template find<2>(&strides[0])*J;
                    auto dof_xx=(geo_map::layout_t::find<1>(&strides[0]))*II+geo_map::layout_t::template find<2>(&strides[0])*JJ;
                    //sum the contribution from elem i-1 on the opposite face
                    eval(out(row+dof_x, col+dof_xx)) += eval(out(i-1
                                                                 , row+(dof_x+geo_map::layout_t::template find<0>(&strides[0])*(bd_dim-1))
                                                                 , col+(dof_xx+geo_map::layout_t::template find<0>(&strides[0])*(bd_dim-1))));

                    auto dof_y=geo_map::layout_t::template find<0>(&strides[0])*I+geo_map::layout_t::template find<2>(&strides[0])*J;
                    auto dof_yy=geo_map::layout_t::template find<0>(&strides[0])*II+geo_map::layout_t::template find<2>(&strides[0])*JJ;
                    //sum the contribution from elem j-1 on the opposite face
                    eval(out(row+dof_y, col+dof_yy)) += eval(out(j-1
                                                     , row+(dof_y+geo_map::layout_t::template find<1>(&strides[0])*bd_dim)
                                                     , col+(dof_yy+geo_map::layout_t::template find<1>(&strides[0])*bd_dim) ));

                    // auto dof_z=fe::layout_t::find<0>(&strides[0])*I+fe::layout_t::find<1>(&strides[0])*J;
                    // auto dof_zz=fe::layout_t::find<0>(&strides[0])*II+fe::layout_t::find<1>(&strides[0])*JJ;
                    // //sum the contribution from elem k-1 on the opposite face
                    // eval(out(row+dof_z, col+dof_zz)) += eval(out(k-1
                    //                                  , row+(dof_z+fe::layout_t::find<2>(&strides[0])*bd_dim)
                    //                                  , col+(dof_zz+fe::layout_t::find<2>(&strides[0])*bd_dim)));

                }
        }
    };
    // [assembly]
} // namespace functors
// } // namespace gridtools
