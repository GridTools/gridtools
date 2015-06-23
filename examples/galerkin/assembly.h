#pragma once

#include <gridtools.h>
#include <stencil-composition/backend.h>
#include <stencil-composition/interval.h>
#include <stencil-composition/make_computation.h>
#include "intrepid.h"

/*
  @file
  @brief This file shows a possible usage of the extension to storages with more than 3 space dimensions.

  We recall that the space dimensions simply identify the number of indexes/strides required to access
  a contiguous chunck of storage. The number of space dimensions is fully arbitrary.

  In particular, we show how to perform a nested inner loop on the extra dimension(s). Possible scenarios
  where this can be useful could be:
  * when dealing with arbitrary order integration of a field in the cells.
  * when we want to implement a discretization scheme involving integrals (like all Galerkin-type discretizations, i.e. continuous/discontinuous finite elements, isogeometric analysis)
  * if we discretize an equation defined on a manifold with more than 3 dimensions (e.g. space-time)
  * if we want to implement coloring schemes, or access the grid points using exotic (but 'regular') patterns

  In this example we suppose that we aim at projecting a field 'f' on a finite elements space. To each
  i,j,k point corresponds an element (we can e.g. suppose that the i,j,k, nodes are the low-left corner).
  We suppose that the following (4-dimensional) quantities are provided (replaced with stubs)
  * The basis and test functions phi and psi respectively, evaluated on the quadrature points of the
  reference element
  * The Jacobian of the finite elements transformation (from the reference to the current configurations)
  , also evaluated in the quadrature points
  * The quadrature nodes/quadrature rule

  With this information we perform the projection (i.e. perform an integral) by looping on the
  quadrature points in an innermost loop, with stride given by the layout_map (I*J*K in this case).

  In this example we introduce also another syntactic element in the high level expression: the operator exclamation mark (!). This operator prefixed to a placeholder means that the corresponding storage index is not considered, and only the offsets are used to get the absolute address. This allows to perform operations which are not stencil-like. It is used in this case to address the basis functions values.
*/
#ifdef CXX11_ENABLED

using namespace gridtools;
using namespace enumtype;
using namespace expressions;

namespace assembly{

    typedef gridtools::interval<level<0,-1>, level<1,-1> > x_interval;
    typedef gridtools::interval<level<0,-2>, level<1,1> > axis;

    /** updates the values of the Jacobian matrix. The Jacobian matrix, component (i,j) in the quadrature point q, is computed given the geometric map discretization as \f$ J(i,j,q)=\sum_k\frac{\partial \phi_i(x_k,q)}{\partial x_j} x_k \f$
        where x_k are the points in the geometric element*/
    template <typename ReferenceSpace>
    struct update{
        typedef accessor<0, range<0,0,0,0> , 4> const grid_points;
        typedef accessor<1, range<0,0,0,0> , 4> jac;
        typedef accessor<2, range<0,0,0,0> , 4> const dphi;
        typedef boost::mpl::vector< grid_points, jac, dphi> arg_list;

        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_interval) {

            for(short_t icoor=0; icoor<3; ++icoor)
                for(short_t jcoor=0; jcoor<3; ++jcoor)
                    for(short_t iter_quad=0; iter_quad<ReferenceSpace::s_nb_quad_pt; ++iter_quad)
                        for (int_t iterNode=0; iterNode < ReferenceSpace::s_dimension; ++iterNode)
                        {//reduction/gather
                            jac(i+icoor, j+jcoor, qp+iter_quad) += grid_point(i+iterNode, d+icoord) * dphi(i+iterNode, d+jcoord, qp+iter_quad);
                        }

            }
    };

    template <typename ReferenceFESpace1, typename ReferenceGeoSpace2>
    struct integration {
        typedef accessor<0, range<0,0,0,0> , 2> const phi;
        typedef accessor<1, range<0,0,0,0> , 2> const psi;//how to detect when index is wrong??
        typedef accessor<2, range<0,0,0,0> , 4> const jac;
        typedef accessor<3, range<0,0,0,0> > const f;
        typedef accessor<4, range<0,0,0,0> > result;
        typedef boost::mpl::vector<phi, psi, jac, f, result> arg_list;
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
            for(short_t I=0; I<ReferenceFESpace1::n_dofs; ++I)
                for(short_t J=0; J<ReferenceFESpace2::n_dofs; ++J)
                    for(short_t q=0; q<2; ++q)
                        eval(result(I)) +=
                            eval(!phi(i+I,qp+q)*!psi(i+J,qp+q) * jac(qp+q)*f()/8;

        }
    };

    std::ostream& operator<<(std::ostream& s, integration const) {
        return s << "integration";
    }

    bool test(uint_t d1, uint_t d2, uint_t d3){

#ifdef CUDA_EXAMPLE
#define BACKEND backend<Cuda, Block >
#else
#ifdef BACKEND_BLOCK
#define BACKEND backend<Host, Block >
#else
#define BACKEND backend<Host, Naive >
#endif
#endif
        //                      dims  x y z  qp
        //                   strides  1 x xy xyz
        typedef gridtools::layout_map<3,2, 1, 0> layout_jacobian_t;
        typedef gridtools::layout_map<2,1,0> layout_t;
        typedef gridtools::layout_map<1,0> layout_basis_func_t;
        typedef gridtools::BACKEND::storage_type<float_type, layout_t >::type storage_type;
        //I might want to store the jacobian as a temporary storage (will use less memory but
        //not reusing when I need the Jacobian for multiple things).
        typedef gridtools::BACKEND::storage_type<float_type, layout_jacobian_t >::type jacobian_type;
        typedef gridtools::BACKEND::storage_type<float_type, layout_basis_func_t >::type basis_func_type;

        /**I have to define here the placeholders to the storages used: the temporary storages get internally managed, while
           non-temporary ones must be instantiated by the user below. In this example all the storages are non-temporaries.*/
        typedef arg<0, basis_func_type > p_phi;
        typedef arg<1, basis_func_type > p_psi;
        typedef arg<2, jacobian_type > p_jac;
        typedef arg<3, storage_type > p_f;
        typedef arg<4, storage_type > p_result;

        typedef boost::mpl::vector<p_phi, p_psi, p_jac, p_f, p_result> accessor_list;

        //the above storage constructors are setting up the storages without allocating the space (might want to change this?). We do it now.
        typedef intrepid<cell_topology<topology::cartesian<layout_t> > , basis_type_policy<float_type, integration_type, 1> , 4> ref_FE;
        typedef intrepid<float_type, 3, 4> ref_GEO;
        ref_FE finite_element;
        ref_FE finite_element2;
        finite_element .initialize_vairables(Intrepid::OPERATOR_VALUE);
        finite_element2.initialize_vairables(Intrepid::OPERATOR_VALUE);

        integration_type jac(d1,d2,d3,ref_FE.nb_quad_pts());

        storage_type f(d1, d2, d3, (float_type)1.3, "f");

        storage_type result(d1, d2, d3, (float_type)0., "result");

        gridtools::domain_type<accessor_list> domain(boost::fusion::make_vector(&finite_element.phi(), &finite_element2.phi(), &jac, &f, &result));

        /**
           - Definition of the physical dimensions of the problem.
           The coordinates constructor takes the horizontal plane dimensions,
           hile the vertical ones are set according the the axis property soon after
        */
        uint_t di[5] = {1, 1, 1, d1-3, d1};
        uint_t dj[5] = {1, 1, 1, d2-3, d2};
        gridtools::coordinates<axis> coords(di,dj);
        coords.value_list[0] = 0;
        coords.value_list[1] = d3-2;



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
                    make_esf<update<ref_GEO> >( p_grid_points(), p_jac(), p_dphi()),
                    make_esf<integration<ref_FE, ref_FE> >(p_phi(), p_psi(), p_jac(), p_f(), p_result())
                    ),
                domain, coords);

        fe_comp->ready();
        fe_comp->steady();
        fe_comp->run();
        fe_comp->finalize();

        result.print();

        return success;
    }

}; //namespace assembly
#endif //CXX11_ENABLED
