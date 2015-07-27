#pragma once

#include <stencil-composition/make_computation.hpp>
// [includes]
#include "basis_functions.h"
#include "intrepid.h"
// [includes]
#ifdef CXX11_ENABLED

// [namespaces]
using namespace gridtools;
using namespace enumtype;
using namespace expressions;
// [namespaces]

typedef gridtools::interval<gridtools::level<0,-1>, gridtools::level<1,-1> > x_interval;
typedef gridtools::interval<gridtools::level<0,-2>, gridtools::level<1,1> > axis;

// [storage_types]
template <typename Geometry>
struct assembly{
    using geo_map=typename Geometry::geo_map;
    using cub=typename Geometry::cub;
    //                      dims  x y z  qp
    //                   strides  1 x xy xyz
    using storage_type=storage_t<gridtools::layout_map<0,1,2,3> >;
    using jacobian_type=storage_t<gridtools::layout_map<0,1,2,3,4,5> >;
    using grid_type=storage_t<gridtools::layout_map<0,1,2,3,4> >;
    static const int_t edge_points=geo_map::hypercube_t::template boundary_w_dim<1>::n_points::value;
// [storage_types]


    /**I have to define here the placeholders to the storages used: the temporary storages get internally managed, while
       non-temporary ones must be instantiated by the user. In this example all the storages are non-temporaries.*/
    typedef arg<0, grid_type >       p_grid_points;
    typedef arg<1, jacobian_type >   p_jac;
    typedef arg<2, typename Geometry::weights_storage_t >   p_weights;
    typedef arg<3, storage_type >   p_jac_det;
    typedef arg<4, jacobian_type >   p_jac_inv;
    typedef arg<5, typename Geometry::grad_storage_t > p_dphi;
    static const uint_t size=5;

    // typedef boost::mpl::vector<p_grid_points, p_jac, // const
    //                            p_weights, p_jac_det, p_jac_inv, // const
    //                            p_dphi// , p_stiffness , p_assembled_stiffness
    //                            > accessor_list;

// [private members]
    uint_t m_d1, m_d2, m_d3;
private:

    Geometry // const
    & m_fe_backend;
    grid_type m_grid;
    jacobian_type m_jac;
    storage_type m_jac_det;
    jacobian_type m_jac_inv;

public:

    assembly(Geometry // const
             & fe_backend_, uint_t d1, uint_t d2, uint_t d3 ): m_fe_backend(fe_backend_)
                                                             , m_d1(d1)
                                                             , m_d2(d2)
                                                             , m_d3(d3)
                                                             , m_grid(d1, d2, d3, geo_map::basisCardinality, 3)
                                                             , m_jac(d1, d2, d3, cub::numCubPoints, 3, 3)
                                                             , m_jac_det(d1, d2, d3, cub::numCubPoints)
                                                             , m_jac_inv(d1, d2, d3, cub::numCubPoints, 3, 3)
        {        }

    jacobian_type const& get_jac() const {return m_jac;}
    storage_type const& get_jac_det() const {return m_jac_det;}
    jacobian_type const& get_jac_inv() const {return m_jac_inv;}
    grid_type const& get_grid() const {return m_grid;}
    grid_type& grid() {return m_grid;}

    template <typename ... MPLList>
    gridtools::domain_type< boost::mpl::vector<p_grid_points, p_jac,
                                               p_weights, p_jac_det, p_jac_inv,
                                               p_dphi, MPLList ...> >
    domain(typename MPLList::storage_type& ...  storages_ ){
        return domain_type<boost::mpl::vector<p_grid_points, p_jac,
                                               p_weights, p_jac_det, p_jac_inv,
                                               p_dphi, MPLList ...> >(boost::fusion::make_vector(&m_grid, &m_jac, &m_fe_backend.cub_weights(), &m_jac_det, &m_jac_inv, &m_fe_backend.local_gradient(), &storages_ ...));
               }
// [private members]

    // [update_jac]
    /** updates the values of the Jacobian matrix. The Jacobian matrix, component (i,j) in the quadrature point q, is computed given the geometric map discretization as \f$ J(i,j,q)=\sum_k\frac{\partial \phi_i(x_k,q)}{\partial x_j} x_k \f$
        where x_k are the points in the geometric element*/
    struct update_jac{
        typedef accessor<0, range<0,0,0,0> , 5> const grid_points;
        typedef accessor<1, range<0,0,0,0> , 6> jac;
        typedef accessor<2, range<0,0,0,0> , 3> const dphi;
        typedef boost::mpl::vector< grid_points, jac, dphi> arg_list;

        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_interval) {
            Dimension<4>::Index qp;
            Dimension<5>::Index dimx;
            Dimension<6>::Index dimy;

            for(short_t icoor=0; icoor< 3; ++icoor)
            {
                for(short_t jcoor=0; jcoor< 3; ++jcoor)
                {
                    for(short_t iter_quad=0; iter_quad< cub::numCubPoints/*quad_pts*/; ++iter_quad)
                    {
                        eval( jac(dimx+icoor, dimy+jcoor, qp+iter_quad) )=0.;
                                for (int_t iterNode=0; iterNode < geo_map::basisCardinality ; ++iterNode)
                                {//reduction/gather
                                    // std::cout<<eval(grid_points(Dimension<4>(iterNode), Dimension<5>(icoor)))<< " * "<<eval(!dphi(iterNode, iter_quad, jcoor));
                                    eval( jac(dimx+icoor, dimy+jcoor, qp+iter_quad) ) += eval(grid_points(Dimension<4>(iterNode), Dimension<5>(icoor)) * !dphi(iterNode, iter_quad, jcoor) );
                                    // std::cout<<" = "<<eval(jac(dimx+icoor, dimy+jcoor, qp+iter_quad))<<std::endl;
                                }
                                // std::cout<<" eventually= "<<eval(jac(dimx+icoor, dimy+jcoor, qp+iter_quad))<<std::endl;
                    }
                    // std::cout<<std::endl;
                }
            }
        }
    };
    // [update_jac]


    //! [det]
    /** updates the values of the Jacobian matrix. The Jacobian matrix, component (i,j) in the quadrature point q, is computed given the geometric map discretization as \f$ J(i,j,q)=\sum_k\frac{\partial \phi_i(x_k,q)}{\partial x_j} x_k \f$
        where x_k are the points in the geometric element*/
    struct det{
        using jac = accessor<0, range<0,0,0,0> , 6> const;
        using jac_det =  accessor<1, range<0,0,0,0> , 4>;
        using arg_list= boost::mpl::vector< jac, jac_det > ;

        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_interval) {
            Dimension<4>::Index qp;
            Dimension<5>::Index dimx;
            Dimension<6>::Index dimy;

            for(short_t q=0; q< cub::numCubPoints; ++q)
            {
                eval( jac_det(qp+q) )= eval( //!weight(q,0,0)*
                                            (
                    jac(        qp+q)*jac(dimx+1, dimy+1, qp+q)*jac(dimx+2, dimy+2, qp+q) +
                    jac(dimx+1, qp+q)*jac(dimx+2, dimy+1, qp+q)*jac(dimy+2,         qp+q) +
                    jac(dimy+1, qp+q)*jac(dimx+1, dimy+2, qp+q)*jac(dimx+2,         qp+q) -
                    jac(dimy+1, qp+q)*jac(dimx+1,         qp+q)*jac(dimx+2, dimy+2, qp+q) -
                    jac(        qp+q)*jac(dimx+2, dimy+1, qp+q)*jac(dimx+1, dimy+2, qp+q) -
                    jac(dimy+2, qp+q)*jac(dimx+1, dimy+1, qp+q)*jac(dimx+2,         qp+q)
                                                ));
            }
        }
    };
    //! [det]

    //! [inv]
    struct inv{
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
            Dimension<4>::Index qp;
            using dimx=Dimension<5>;
            using dimy=Dimension<6>;
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
            for(short_t q=0; q< cub::numCubPoints/*quad_pts*/; ++q)
            {
                alias<a_, Dimension<4> > a(q);
                alias<b_, Dimension<4> > b(q);
                alias<c_, Dimension<4> > c(q);
                alias<d_, Dimension<4> > d(q);
                alias<e_, Dimension<4> > e(q);
                alias<f_, Dimension<4> > f(q);
                alias<g_, Dimension<4> > g(q);
                alias<h_, Dimension<4> > h(q);
                alias<i_, Dimension<4> > i(q);

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
    struct assembly_f {

        using in=accessor<0, range<-1,0,-1,0> , 5> const;
        using out=accessor<1, range<0,0,0,0> , 5> ;
        using arg_list=boost::mpl::vector<in, out> ;
        using quad=Dimension<4>;

        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_interval) {
            x::Index i;
            y::Index j;
            z::Index k;
            Dimension<4>::Index row;
            Dimension<5>::Index col;

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

}; //struct assembly

template<typename GEO>
const int_t assembly< GEO >::edge_points;

#endif //CXX11_ENABLED
