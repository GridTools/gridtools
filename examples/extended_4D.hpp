#pragma once

#include <gridtools.hpp>
#include <stencil-composition/backend.hpp>
#include <stencil-composition/interval.hpp>
#include <stencil-composition/make_computation.hpp>

/**
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

    struct integration {
        typedef accessor<0, range<-1, 1, -1, 1> , 4> const phi;
        typedef accessor<1, range<-1, 1, -1, 1> , 4> const psi;//how to detect when index is wrong??
        typedef accessor<2, range<-1, 1, -1, 1> , 4> const jac;
        typedef accessor<3, range<-1, 1, -1, 1> > const f;
        typedef accessor<4, range<-1, 1, -1, 1> > result;
        typedef boost::mpl::vector<phi, psi, jac, f, result> arg_list;
        using quad=dimension<4>;
        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_interval) {
            x::Index i;
            y::Index j;
            z::Index k;
            quad::Index qp;
            //projection of f on a (e.g.) P1 FE space:
            //loop on quadrature nodes, and on nodes of the P1 element (i,j,k) with i,j,k\in {0,1}
            //computational complexity in the order of  {(I) x (J) x (K) x (i) x (j) x (k) x (nq)}
            for(short_t I=0; I<2; ++I)
                for(short_t J=0; J<2; ++J)
                    for(short_t K=0; K<2; ++K)
                        for(short_t q=0; q<2; ++q)
                            eval(result(I,J,K)) +=
                                eval(!phi(i+I,j+J,k+K,qp+q)*!psi(qp+q)             *jac(qp+q)*f() +
                                     !phi(i+I,j+J,k+K,qp+q)*!psi(i+1, qp+q)        *jac(qp+q)*f(i+1) +
                                     !phi(i+I,j+J,k+K,qp+q)*!psi(j+1, qp+q)        *jac(qp+q)*f(j+1) +
                                     !phi(i+I,j+J,k+K,qp+q)*!psi(k+1, qp+q)        *jac(qp+q)*f(k+1) +
                                     !phi(i+I,j+J,k+K,qp+q)*!psi(i+1, j+1, qp+q)   *jac(qp+q)*f(i+1, j+1) +
                                     !phi(i+I,j+J,k+K,qp+q)*!psi(i+1, k+1, qp+q)   *jac(qp+q)*f(i+1, k+1) +
                                     !phi(i+I,j+J,k+K,qp+q)*!psi(j+1,k+1, qp+q)    *jac(qp+q)*f(j+1,k+1) +
                                     !phi(i+I,j+J,k+K,qp+q)*!psi(i+1,j+1,k+1, qp+q)*jac(qp+q)*f(i+1,j+1,k+1))
                                /8;

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
        typedef gridtools::layout_map<3,2, 1, 0> layout4_t;
        typedef gridtools::layout_map<2,1,0> layout_t;
        typedef gridtools::BACKEND::storage_type<float_type, layout_t >::type storage_type;
        typedef gridtools::BACKEND::storage_type<float_type, layout4_t >::type integration_type;

        typedef arg<0, integration_type > p_phi;
        typedef arg<1, integration_type > p_psi;
        typedef arg<2, integration_type > p_jac;
        typedef arg<3, storage_type > p_f;
        typedef arg<4, storage_type > p_result;

        typedef boost::mpl::vector<p_phi, p_psi, p_jac, p_f, p_result> accessor_list;

        uint_t nbQuadPt=2;//referenceFE_Type::nbQuadPt;
        uint_t b1=2;
        uint_t b2=2;
        uint_t b3=2;
        //basis functions available in a 2x2x2 cell, because of P1 FE
        integration_type phi(b1,b2,b3,nbQuadPt);
        integration_type psi(b1,b2,b3,nbQuadPt);

        //I might want to treat it as a temporary storage (will use less memory but constantly copying back and forth)
        //Or alternatively computing the values on the quadrature points on the GPU
        integration_type jac(d1,d2,d3,nbQuadPt);

        //the above storage constructors are setting up the storages without allocating the space (might want to change this?). We do it now.
        jac.allocate();
        psi.allocate();
        phi.allocate();

        for(uint_t i=0; i<d1; ++i)
            for(uint_t j=0; j<d2; ++j)
                for(uint_t k=0; k<d3; ++k)
                    for(uint_t q=0; q<nbQuadPt; ++q)
                    {
                        jac(i,j,k,q)=1.+q;
                    }
        for(uint_t i=0; i<b1; ++i)
            for(uint_t j=0; j<b2; ++j)
                for(uint_t k=0; k<b3; ++k)
                    for(uint_t q=0; q<nbQuadPt; ++q)
                    {
                        phi(i,j,k,q)=10.;
                        psi(i,j,k,q)=11.;
                    }
        storage_type f(d1, d2, d3, (float_type)1.3, "f");

        storage_type result(d1, d2, d3, (float_type)0., "result");

        gridtools::domain_type<accessor_list> domain(boost::fusion::make_vector(&phi, &psi, &jac, &f, &result));
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
                make_mss //! \todo all the arguments in the call to make_mss are actually dummy.
                (
                    execute<forward>(),//!\todo parameter used only for overloading purpose?
                    make_esf<integration>(p_phi(), p_psi(), p_jac(), p_f(), p_result())
                    ),
                domain, coords);

        fe_comp->ready();
        fe_comp->steady();
        fe_comp->run();
        fe_comp->finalize();

        bool success(true);
        for(uint_t i=0; i<d1; ++i)
            for(uint_t j=0; j<d2; ++j)
                for(uint_t k=0; k<d3; ++k)
                {
                    if (result(i, j, k)!=((1*1.3*10*11*2*2*2)+(2*1.3*10*11*2*2*2))/((i==2&&j==2)?1:2)/ ((k==0||k==5)?2:1) /(((i==1||i==3)&&(j==1||j==3))?2:1)*((i==0||i==4||j==0||j==4)?0:1)) {
                        std::cout << "error in "
                                  << i << ", "
                                  << j << ", "
                                  << k << ": "
                                  << "result = " << result(i, j, k)
                                  << " instead of " << ((1*1.3*10*11*2*2*2)+(2*1.3*10*11*2*2*2))/(((i==1||i==3)&&(j==1||j==3))?2:1)/(((i==1||i==3)&&(j==1||j==3)&&(k==0||k==5))?2:1)*((i==0||i==4||j==0||j==4)?0:1)
                                  << std::endl;
                        success = false;
                    }
                }


        //jac.print();
        //phi.print();
        //psi.print();
        return success;
    }

}; //namespace assembly
#endif //CXX11_ENABLED
