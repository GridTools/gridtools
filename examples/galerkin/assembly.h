#pragma once

// [includes]
#include <stencil-composition/interval.h>
#include <stencil-composition/make_computation.h>
#include "basis_functions.h"

#ifdef CXX11_ENABLED

using namespace gridtools;
using namespace enumtype;
using namespace expressions;

namespace assembly{

    typedef gridtools::interval<level<0,-1>, level<1,-1> > x_interval;
    typedef gridtools::interval<level<0,-2>, level<1,1> > axis;

    /** updates the values of the Jacobian matrix. The Jacobian matrix, component (i,j) in the quadrature point q, is computed given the geometric map discretization as \f$ J(i,j,q)=\sum_k\frac{\partial \phi_i(x_k,q)}{\partial x_j} x_k \f$
        where x_k are the points in the geometric element*/
    struct update{
        typedef accessor<0, range<0,0,0,0> , 4> const grid_points;
        typedef accessor<1, range<0,0,0,0> , 5> jac;
        typedef accessor<2, range<0,0,0,0> , 3> const dphi;
        typedef boost::mpl::vector< grid_points, jac, dphi> arg_list;

        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_interval) {
            Dimension<1>::Index i;
            Dimension<2>::Index j;
            Dimension<3>::Index qp;
            Dimension<4>::Index dimx;
            Dimension<5>::Index dimy;
            Dimension<2>::Index d;

            for(short_t icoor=0; icoor< 3; ++icoor)
                for(short_t jcoor=0; jcoor< 3; ++jcoor)
                    for(short_t iter_quad=0; iter_quad< fe::numNodes/*quad_pts*/; ++iter_quad)
                    {
                        eval( jac(dimx+icoor, dimy+jcoor, qp+iter_quad) )=0.;
                        for (int_t iterNode=0; iterNode < fe::numNodes ; ++iterNode)
                        {//reduction/gather
                            eval( jac(dimx+icoor, dimy+jcoor, qp+iter_quad) ) += eval(grid_points(iterNode, icoor, 0, 0) * !dphi(i+iterNode, d+jcoor, qp+iter_quad) );
                        }
                    }
            }
    };

    template <typename ReferenceFESpace1, typename ReferenceFESpace2>
    struct integration {
        typedef accessor<0, range<0,0,0,0> , 3> const dphi;
        typedef accessor<2, range<0,0,0,0> , 5> const jac;
        typedef accessor<3, range<0,0,0,0> , 3> const f;
        typedef accessor<4, range<0,0,0,0> , 3> result;
        typedef boost::mpl::vector<dphi, jac, f, result> arg_list;
        using quad=Dimension<4>;

        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_interval) {
            x::Index i;
            y::Index j;
            z::Index k;
            quad::Index qp;
            //projection of f on a (e.g.) P1 FE space ReferenceFESpace1:
            //loop on quadrature nodes, and on nodes of the P1 element (i,j,k) with i,j,k\in {0,1}
            //computational complexity in the order of  {(nDOF) x (nDOF) x (nq)}
            for(short_t I=0; I<8; ++I)
                for(short_t J=0; J<8; ++J)
                    for(short_t q=0; q<2; ++q)
                        eval(result(I)) +=
                            eval(!dphi(i+I,qp+q)*!dphi(i+J,qp+q) * jac(qp+q)*f(qp+q))/8;

        }
    };


    template <typename ReferenceFESpace1, typename ReferenceFESpace2>
    struct assembly {
        typedef accessor<0, range<0,0,0,0> , 3> const in;
        typedef accessor<1, range<0,0,0,0> , 3> out;
        typedef boost::mpl::vector<in, out> arg_list;
        using quad=Dimension<4>;

        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_interval) {
            x::Index i;

            //assembly
            for(short_t I=0; I<4; ++I)
            {
                eval(out(i+I)) += eval(in(i+I));
                eval(out(i+I)) += eval(in(i+(I-4)));
            }
        }
    };

    // template<typename T, typename U>
    // std::ostream& operator<<(std::ostream& s, integration<T,U> const) {
    //     return s << "integration";
    // }

    template <typename StorageGradType>
    bool test( StorageGradType& local_gradient, int cub_points ){

        uint_t d1=10;
        uint_t d2=10;
        uint_t d3=1;

        //                      dims  x y z  qp
        //                   strides  1 x xy xyz
        typedef gridtools::layout_map<2,1,0> layout_t;
        typedef gridtools::BACKEND::storage_type<float_type, layout_t >::type storage_type;
        typedef gridtools::layout_map<4,3,2,1,0> layout_jacobian_t;
        typedef gridtools::BACKEND::storage_type<float_type, layout_jacobian_t >::type jacobian_type;
        typedef gridtools::layout_map<3,2,1,0> layout_grid_t;
        typedef gridtools::BACKEND::storage_type<float_type, layout_grid_t >::type grid_type;
        //I might want to store the jacobian as a temporary storage (will use less memory but
        //not reusing when I need the Jacobian for multiple things).

        /**I have to define here the placeholders to the storages used: the temporary storages get internally managed, while
           non-temporary ones must be instantiated by the user below. In this example all the storages are non-temporaries.*/
        // typedef arg<0, basis_func_type > p_phi;
        // typedef arg<1, basis_func_type > p_psi;
        typedef arg<0, grid_type >       p_grid_points;
        typedef arg<1, jacobian_type >   p_jac;
        typedef arg<2, StorageGradType > p_dphi;
        // typedef arg<3, storage_type >    p_f;
        // typedef arg<4, storage_type >    p_result;

        typedef boost::mpl::vector<p_grid_points, p_jac, p_dphi// , p_f, p_result
                                   > accessor_list;

        grid_type grid(d1, d2, fe::numNodes, 3);
        grid.initialize(0.);

        jacobian_type jac(d1, d2, cub_points, 3, 3);
        jac.initialize(0.);

        // storage_type f(d1, d2, d3, (float_type)1.3, "f");
        // f.initialize(0.);

        // storage_type result(d1, d2, d3, (float_type)0., "result");
        // result.initialize(0.);

        gridtools::domain_type<accessor_list> domain(boost::fusion::make_vector(&grid, &jac, &local_gradient));

        /**
           - Definition of the physical dimensions of the problem.
           The coordinates constructor takes the horizontal plane dimensions,
           hile the vertical ones are set according the the axis property soon after
        */
        uint_t di[5] = {1, 1, 1, d1-3, d1};
        uint_t dj[5] = {1, 1, 1, d2-3, d2};
        gridtools::coordinates<axis> coords(di,dj);
        coords.value_list[0] = 0;
        coords.value_list[1] = d3;

#ifdef __CUDACC__
        computation* fe_comp =
#else
            boost::shared_ptr<gridtools::computation> fe_comp =
#endif
            make_computation<gridtools::BACKEND, layout_t>
            (
                make_mss
                (
                    execute<forward>(),
                    make_esf<update>( p_grid_points(), p_jac(), p_dphi())
                    //make_esf<integration<ref_FE, ref_FE> >(p_phi(), p_psi(), p_jac(), p_f(), p_result())
                    ),
                domain, coords);

        fe_comp->ready();
        fe_comp->steady();
        fe_comp->run();
        fe_comp->finalize();

        // result.print();

        return true;
    }

}; //namespace assembly
#endif //CXX11_ENABLED
