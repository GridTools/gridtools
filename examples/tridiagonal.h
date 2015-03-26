#pragma once

#include <gridtools.h>

#include <stencil-composition/backend.h>

#include <stencil-composition/interval.h>
#include <stencil-composition/make_computation.h>

#ifdef USE_PAPI_WRAP
#include <papi_wrap.h>
#include <papi.h>
#endif

/*
  @file This file shows an implementation of the Thomas algorithm, done using stencil operations.

  Important convention: the linear system as usual is represented with 4 vectors: the main diagonal
  (diag), the upper and lower first diagonals (sup and inf respectively), and the right hand side
  (rhs). Note that the dimensions and the memory layout are, for an NxN system
  rank(diag)=N       [xxxxxxxxxxxxxxxxxxxxxxxx]
  rank(inf)=N-1      [0xxxxxxxxxxxxxxxxxxxxxxx]
  rank(sup)=N-1      [xxxxxxxxxxxxxxxxxxxxxxx0]
  rank(rhs)=N        [xxxxxxxxxxxxxxxxxxxxxxxx]
  where x denotes any number and 0 denotes the padding, a dummy value which is not used in
  the algorithm. This choice coresponds to having the same vector index for each row of the matrix.
 */

using gridtools::level;
using gridtools::arg_type;
using gridtools::range;
using gridtools::arg;

namespace tridiagonal{

using namespace gridtools;
using namespace enumtype;

#ifdef CXX11_ENABLED
using namespace expressions;
#endif

// This is the definition of the special regions in the "vertical" direction
typedef gridtools::interval<level<0,1>, level<1,-2> > x_internal;
typedef gridtools::interval<level<0,-1>, level<0,-1> > x_first;
typedef gridtools::interval<level<1,-1>, level<1,-1> > x_last;
typedef gridtools::interval<level<0,-1>, level<1,1> > axis;

#if (defined(CXX11_ENABLED)  && !defined(__CUDACC__ ))
    namespace ex{
        typedef arg_type<0> out;
        typedef arg_type<1> inf; //a
        typedef arg_type<2> diag; //b
        typedef arg_type<3> sup; //c
        typedef arg_type<4> rhs; //d

        static  auto expr_sup=sup{}/(diag{}-sup{z{-1}}*inf{});

        static  auto expr_rhs=(rhs{}-inf{}*rhs{z{-1}})/(diag{}-sup{z{-1}}*inf{});

        static  auto expr_out=rhs{}-sup{}*out{0,0,1};
    }
#endif

struct forward_thomas{
//four vectors: output, and the 3 diagonals
#ifdef CXX11_ENABLED
    typedef arg_type<0> out;
    typedef arg_type<1> inf; //a
    typedef arg_type<2> diag; //b
    typedef arg_type<3> sup; //c
    typedef arg_type<4> rhs; //d
#else
    typedef arg_type<0>::type out;
    typedef arg_type<1>::type inf; //a
    typedef arg_type<2>::type diag; //b
    typedef arg_type<3>::type sup; //c
    typedef arg_type<4>::type rhs; //d
#endif
    typedef boost::mpl::vector<out, inf, diag, sup, rhs> arg_list;

    template <typename Domain>
    GT_FUNCTION
    static void shared_kernel(Domain const& dom) {
#if (defined(CXX11_ENABLED) && (!defined(__CUDACC__ )))
	dom(sup()) =  dom(ex::expr_sup);
	dom(rhs()) =  dom(ex::expr_rhs);
#else
        dom(sup()) = dom(sup())/(dom(diag())-dom(sup(z(-1)))*dom(inf()));
        dom(rhs()) = (dom(rhs())-dom(inf())*dom(rhs(z(-1))))/(dom(diag())-dom(sup(z(-1)))*dom(inf()));
#endif
    }

    template <typename Domain>
    GT_FUNCTION
    static void Do(Domain const & dom, x_internal) {
        shared_kernel(dom);
    }

    template <typename Domain>
    GT_FUNCTION
    static void Do(Domain const & dom, x_last) {
        shared_kernel(dom);
    }

    template <typename Domain>
    GT_FUNCTION
    static void Do(Domain const & dom, x_first) {
        dom(sup()) = dom(sup())/dom(diag());
        dom(rhs()) = dom(rhs())/dom(diag());
    }

};

struct backward_thomas{
#ifdef CXX11_ENABLED
    typedef arg_type<0> out;
    typedef arg_type<1> inf; //a
    typedef arg_type<2> diag; //b
    typedef arg_type<3> sup; //c
    typedef arg_type<4> rhs; //d
#else
    typedef arg_type<0>::type out;
    typedef arg_type<1>::type inf; //a
    typedef arg_type<2>::type diag; //b
    typedef arg_type<3>::type sup; //c
    typedef arg_type<4>::type rhs; //d
#endif
    typedef boost::mpl::vector<out, inf, diag, sup, rhs> arg_list;


    template <typename Domain>
    GT_FUNCTION
    static void shared_kernel(Domain& dom) {
#if (defined(CXX11_ENABLED) && (!defined(__CUDACC__ )))
        dom(out()) = dom(ex::expr_out);
#else
        dom(out()) = dom(rhs())-dom(sup())*dom(out(0,0,1));
#endif
    }

    template <typename Domain>
    GT_FUNCTION
    static void Do(Domain const & dom, x_internal) {
        shared_kernel(dom);
    }

    template <typename Domain>
    GT_FUNCTION
    static void Do(Domain const & dom, x_first) {
        shared_kernel(dom);
    }

    template <typename Domain>
    GT_FUNCTION
    static void Do(Domain const & dom, x_last) {
        dom(out())= dom(rhs());
        // dom(out())= dom(rhs())/dom(diag());
    }
};

std::ostream& operator<<(std::ostream& s, backward_thomas const) {
    return s << "backward_thomas";
}
std::ostream& operator<<(std::ostream& s, forward_thomas const) {
    return s << "forward_thomas";
}


bool solver(uint_t x, uint_t y, uint_t z) {

#ifdef USE_PAPI_WRAP
  int collector_init = pw_new_collector("Init");
  int collector_execute = pw_new_collector("Execute");
#endif

    uint_t d1 = x;
    uint_t d2 = y;
    uint_t d3 = z;

#ifdef CUDA_EXAMPLE
#define BACKEND backend<Cuda, Naive >
#else
#ifdef BACKEND_BLOCK
#define BACKEND backend<Host, Block >
#else
#define BACKEND backend<Host, Naive >
#endif
#endif

    //    typedef gridtools::STORAGE<double, gridtools::layout_map<0,1,2> > storage_type;
    typedef gridtools::layout_map<0,1,2> layout_t;
    typedef gridtools::BACKEND::storage_type<float_type, layout_t >::type storage_type;
    typedef gridtools::BACKEND::temporary_storage_type<float_type, layout_t >::type tmp_storage_type;

     // Definition of the actual data fields that are used for input/output
    //storage_type in(d1,d2,d3,-1, "in"));
    storage_type out(d1,d2,d3,0., "out");
    storage_type inf(d1,d2,d3,-1., "inf");
    storage_type diag(d1,d2,d3,3., "diag");
    storage_type sup(d1,d2,d3,1., "sup");
    storage_type rhs(d1,d2,d3,3., "rhs");
    for(int_t i=0; i<d1; ++i)
        for(int_t j=0; j<d2; ++j)
        {
            rhs(i, j, 0)=4.;
            rhs(i, j, 5)=2.;
        }
// result is 1
    printf("Print OUT field\n");
    out.print();
    printf("Print SUP field\n");
    sup.print();
    printf("Print RHS field\n");
    rhs.print();

    // Definition of placeholders. The order of them reflect the order the user will deal with them
    // especially the non-temporary ones, in the construction of the domain
    typedef arg<0, storage_type > p_inf; //a
    typedef arg<1, storage_type > p_diag; //b
    typedef arg<2, storage_type > p_sup; //c
    typedef arg<3, storage_type > p_rhs; //d
    typedef arg<4, storage_type > p_out;

    // An array of placeholders to be passed to the domain
    // I'm using mpl::vector, but the final API should look slightly simpler
    typedef boost::mpl::vector<p_inf, p_diag, p_sup, p_rhs, p_out> arg_type_list;

    // construction of the domain. The domain is the physical domain of the problem, with all the physical fields that are used, temporary and not
    // It must be noted that the only fields to be passed to the constructor are the non-temporary.
    // The order in which they have to be passed is the order in which they appear scanning the placeholders in order. (I don't particularly like this)
    gridtools::domain_type<arg_type_list> domain
        (boost::fusion::make_vector(&inf, &diag, &sup, &rhs, &out));

    // Definition of the physical dimensions of the problem.
    // The constructor takes the horizontal plane dimensions,
    // while the vertical ones are set according the the axis property soon after
    // gridtools::coordinates<axis> coords(2,d1-2,2,d2-2);
    uint_t di[5] = {0, 0, 0, d1-1, d1};
    uint_t dj[5] = {0, 0, 0, d2-1, d2};

    gridtools::coordinates<axis> coords(di, dj);
    coords.value_list[0] = 0;
    coords.value_list[1] = d3-1;

    /*
      Here we do lot of stuff
      1) We pass to the intermediate representation ::run function the description
      of the stencil, which is a multi-stage stencil (mss)
      The mss includes (in order of execution) a laplacian, two fluxes which are independent
      and a final step that is the out_function
      2) The logical physical domain with the fields to use
      3) The actual domain dimensions
     */


// \todo simplify the following using the auto keyword from C++11
#ifdef __CUDACC__
    gridtools::computation* forward_step =
#else
        boost::shared_ptr<gridtools::computation> forward_step =
#endif
      gridtools::make_computation<gridtools::BACKEND, layout_t>
        (
            gridtools::make_mss // mss_descriptor
            (
                execute<forward>(),
                gridtools::make_esf<forward_thomas>(p_out(), p_inf(), p_diag(), p_sup(), p_rhs()) // esf_descriptor
                ),
            domain, coords
            );


// \todo simplify the following using the auto keyword from C++11
#ifdef __CUDACC__
	gridtools::computation* backward_step =
#else
	       boost::shared_ptr<gridtools::computation> backward_step =
#endif
      gridtools::make_computation<gridtools::BACKEND, layout_t>
      (
            gridtools::make_mss // mss_descriptor
            (
                execute<backward>(),
                gridtools::make_esf<backward_thomas>(p_out(), p_inf(), p_diag(), p_sup(), p_rhs()) // esf_descriptor
                ),
            domain, coords
	  ) ;

    forward_step->ready();
    forward_step->steady();


    forward_step->run();

    forward_step->finalize();

    // printf("Print OUT field (forward)\n");
    // out.print();
    // printf("Print SUP field (forward)\n");
    // sup.print();
    // printf("Print RHS field (forward)\n");
    // rhs.print();

    backward_step->ready();
    backward_step->steady();

    backward_step->run();

    backward_step->finalize();


    //    in.print();
    printf("Print OUT field\n");
    out.print();
    printf("Print SUP field\n");
    sup.print();
    printf("Print RHS field\n");
    rhs.print();
    //    lap.print();

    return (out(0,0,0) + out(0,0,1) + out(0,0,2) + out(0,0,3) + out(0,0,4) + out(0,0,5) >6-1e-10) &&
      (out(0,0,0) + out(0,0,1) + out(0,0,2) + out(0,0,3) + out(0,0,4) + out(0,0,5) <6+1e-10);
}
}//namespace tridiagonal
