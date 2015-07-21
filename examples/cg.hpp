#pragma once

#include <gridtools.hpp>

#include <stencil-composition/backend.hpp>

#include <stencil-composition/interval.hpp>
#include <stencil-composition/make_computation.hpp>

#ifdef USE_PAPI_WRAP
#include <papi_wrap.hpp>
#include <papi.hpp>
#endif

/*
  @file This file shows an implementation of the Conjugate gradient, done using stencil operations.
 */

using gridtools::level;
using gridtools::accessor;
using gridtools::range;
using gridtools::arg;

namespace cg{

using namespace gridtools;
using namespace enumtype;

#ifdef CXX11_ENABLED
using namespace expressions;
#endif

// This is the definition of the special regions in the "vertical" direction
// What does this do? [???]
typedef gridtools::interval<level<0,-1>, level<1,-1> > x_interval;
typedef gridtools::interval<level<0,-1>, level<1,1> > axis;

struct vector_product{
    typedef accessor<0> out;
    typedef accessor<1> a;
    typedef accessor<2> b;
    typedef boost::mpl::vector<out, a, b> arg_list;

    template <typename Domain>
    GT_FUNCTION
    static void Do(Domain const & dom, x_interval) {
        dom(out()) = dom(a()) * dom(b());
    }
};

bool solver(uint_t x, uint_t y, uint_t z) {

    uint_t d1 = x;
    uint_t d2 = y;
    uint_t d3 = z;

#ifdef CUDA_EXAMPLE
#define BACKEND backend<Cuda, Block >
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
    storage_type a(d1,d2,d3,3., "diag");//TODO
    storage_type b(d1,d2,d3,5., "rhs");//TODO
    for(int_t i=0; i<d1; ++i)
        for(int_t j=0; j<d2; ++j)
        {
            a(i, j, 0)=i*d1 + j;
            b(i, j, 0)=i*d1 + j;
            out(i,j,0)=0.;
        }

    printf("Print OUT field\n");
    out.print();
    printf("Print A field\n");
    a.print();
    printf("Print B field\n");
    b.print();

    // Definition of placeholders. The order of them reflect the order the user will deal with them
    // especially the non-temporary ones, in the construction of the domain
    typedef arg<0, storage_type > p_a; //a
    typedef arg<1, storage_type > p_b; //b
    typedef arg<2, storage_type > p_out;

    // An array of placeholders to be passed to the domain
    // I'm using mpl::vector, but the final API should look slightly simpler
    typedef boost::mpl::vector<p_a, p_b, p_out> accessor_list;

    // construction of the domain. The domain is the physical domain of the problem, with all the physical fields that are used, temporary and not
    // It must be noted that the only fields to be passed to the constructor are the non-temporary.
    // The order in which they have to be passed is the order in which they appear scanning the placeholders in order. (I don't particularly like this)
    gridtools::domain_type<accessor_list> domain
        (boost::fusion::make_vector(&a, &b, &out));

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
                gridtools::make_esf<vector_product>(p_out(), p_a(), p_b()) // esf_descriptor
                ),
            domain, coords
            );


    forward_step->ready();
    forward_step->steady();
    forward_step->run();
    forward_step->finalize();


    printf("Print OUT field\n");
    out.print();
    printf("Print A field\n");
    a.print();
    printf("Print B field\n");
    b.print();

    float sum = 0;
    #pragma omp parallel for reduction(+:sum)
    for(int_t i=0; i<d1; ++i) {
        for(int_t j=0; j<d2; ++j)
        {
            sum += out(i,j,0);
        }
    }
    printf("Sum is: %f\n", sum);

    

    return (out(0,0,0) + out(0,0,1) + out(0,0,2) + out(0,0,3) + out(0,0,4) + out(0,0,5) >6-1e-10) &&
      (out(0,0,0) + out(0,0,1) + out(0,0,2) + out(0,0,3) + out(0,0,4) + out(0,0,5) <6+1e-10);
}
}//namespace tridiagonal
