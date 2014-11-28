
#pragma once

#include <gridtools.h>
#ifdef CUDA_EXAMPLE
#include <stencil-composition/backend_cuda.h>
#else
#include <stencil-composition/backend_host.h>
#endif

#include <boost/timer/timer.hpp>
#include <boost/fusion/include/make_vector.hpp>

#ifdef USE_PAPI_WRAP
#include <papi_wrap.h>
#include <papi.h>
#endif

/*
  This file shows an implementation of the "copy" stencil, simple copy of one field done on the backend
 */

using gridtools::level;
using gridtools::arg_type;
using gridtools::range;
using gridtools::arg;

using namespace gridtools;
using namespace enumtype;


namespace copy_stencil{
// This is the definition of the special regions in the "vertical" direction
typedef gridtools::interval<level<0,-1>, level<1,-1> > x_interval;
typedef gridtools::interval<level<0,-2>, level<1,1> > axis;

// These are the stencil operators that compose the multistage stencil in this test
struct copy_functor {
#ifdef CXX11_ENABLED
    typedef arg_extend<arg_type<0>, 1>::type in;
    typedef boost::mpl::vector<in> arg_list;
    typedef Dimension<3> time;
#else
    typedef const arg_type<0> in;
    typedef arg_type<1> out;
    typedef boost::mpl::vector<in,out> arg_list;
#endif
    /* static const auto expression=in(1,0,0)-out(); */

    template <typename Evaluation>
    GT_FUNCTION
    static void Do(Evaluation const & eval, x_interval) {
#ifdef CXX11_ENABLED
        eval(in(1))
#else
        eval(out())
#endif
=eval(in());
      }
};

/*
 * The following operators and structs are for debugging only
 */
std::ostream& operator<<(std::ostream& s, copy_functor const) {
    return s << "copy_functor";
}

void handle_error(int_t)
{std::cout<<"error"<<std::endl;}

bool test(uint_t x, uint_t y, uint_t z) {

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
    //                      dims  z y x
    //                   strides xy x 1
    typedef gridtools::layout_map<2,1,0> layout_t;
    typedef gridtools::BACKEND::storage_type<float_type, layout_t >::type storage_type;
    //typedef storage_type::basic_type integrator_type;
    /* typedef extend<storage_type::basic_type, 2> integrator_type; */
#ifdef CXX11_ENABLED
    typedef extend_width<storage_type::basic_type, 0>  extended_type;
    typedef extend_dim<extended_type, extended_type>  integrator_type;
#endif
    //out.print();

    // Definition of placeholders. The order of them reflect the order the user will deal with them
    // especially the non-temporary ones, in the construction of the domain
#ifdef CXX11_ENABLED
    typedef arg<0, integrator_type > p_in;
    typedef boost::mpl::vector<p_in> arg_type_list;
#else
    typedef arg<0, storage_type> p_in;
    typedef arg<1, storage_type> p_out;
    // An array of placeholders to be passed to the domain
    // I'm using mpl::vector, but the final API should look slightly simpler
    typedef boost::mpl::vector<p_in, p_out> arg_type_list;
#endif
    /* typedef arg<1, integrator_type > p_out; */


    // Definition of the actual data fields that are used for input/output
#ifdef CXX11_ENABLED
    integrator_type in(d1,d2,d3,-3.5);
    integrator_type::original_storage  out(d1,d2,d3,1.5);
#else
    storage_type in(d1,d2,d3,-3.5);
    storage_type out(d1,d2,d3,1.5);
#endif

    for(uint_t i=0; i<d1; ++i)
        for(uint_t j=0; j<d2; ++j)
            for(uint_t k=0; k<d3; ++k)
            {
#ifdef CXX11_ENABLED
                out(i, j, k)=i+j+k;
#else
                in(i, j, k)=i+j+k;
#endif
            }

#ifdef CXX11_ENABLED
    in.push_front(out.m_data);
#endif

    // construction of the domain. The domain is the physical domain of the problem, with all the physical fields that are used, temporary and not
    // It must be noted that the only fields to be passed to the constructor are the non-temporary.
    // The order in which they have to be passed is the order in which they appear scanning the placeholders in order. (I don't particularly like this)
#ifdef CXX11_ENABLED
    gridtools::domain_type<arg_type_list> domain
      (boost::fusion::make_vector(&in));
#else
    gridtools::domain_type<arg_type_list> domain
      (boost::fusion::make_vector(&in, &out));
#endif
    // Definition of the physical dimensions of the problem.
    // The constructor takes the horizontal plane dimensions,
    // while the vertical ones are set according the the axis property soon after
    // gridtools::coordinates<axis> coords(2,d1-2,2,d2-2);
    uint_t di[5] = {0, 0, 0, d1, d1};
    uint_t dj[5] = {0, 0, 0, d2, d2};

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

#ifdef USE_PAPI
int event_set = PAPI_NULL;
int retval;
long long values[1] = {-1};


/* Initialize the PAPI library */
retval = PAPI_library_init(PAPI_VER_CURRENT);
if (retval != PAPI_VER_CURRENT) {
  fprintf(stderr, "PAPI library init error!\n");
  exit(1);
}

if( PAPI_create_eventset(&event_set) != PAPI_OK)
    handle_error(1);
if( PAPI_add_event(event_set, PAPI_FP_INS) != PAPI_OK) //floating point operations
    handle_error(1);
#endif

#ifdef USE_PAPI_WRAP
    pw_start_collector(collector_init);
#endif

// \todo simplify the following using the auto keyword from C++11
#ifdef __CUDACC__
    gridtools::computation* copy =
#else
        boost::shared_ptr<gridtools::computation> copy =
#endif
      gridtools::make_computation<gridtools::BACKEND, layout_t>
        (
            gridtools::make_mss // mss_descriptor
            (
                execute<forward>(),
                gridtools::make_esf<copy_functor>(p_in() // esf_descriptor
#ifndef CXX11_ENABLED
                                                  ,p_out()
#endif
                    )
                ),
            domain, coords
            );

    copy->ready();

    copy->steady();
    domain.clone_to_gpu();

#ifdef USE_PAPI_WRAP
    pw_stop_collector(collector_init);
#endif

    /* boost::timer::cpu_timer time; */
#ifdef USE_PAPI
if( PAPI_start(event_set) != PAPI_OK)
    handle_error(1);
#endif
#ifdef USE_PAPI_WRAP
    pw_start_collector(collector_execute);
#endif
    copy->run();

#ifdef USE_PAPI
double dummy=0.5;
double dummy2=0.8;
if( PAPI_read(event_set, values) != PAPI_OK)
    handle_error(1);
printf("%f After reading the counters: %lld\n", dummy, values[0]);
PAPI_stop(event_set, values);
#endif
#ifdef USE_PAPI_WRAP
    pw_stop_collector(collector_execute);
#endif
    /* boost::timer::cpu_times lapse_time = time.elapsed(); */

    copy->finalize();

//#ifdef CUDA_EXAMPLE
    //out.data().update_cpu();
//#endif
#define NX 511
#define NY 511
#define NZ 59

#ifdef USE_PAPI_WRAP
    pw_print();
#endif

#ifdef CXX11_ENABLED
    in.print_value(NX,NY,0);
    in.print_value(NX,0,NZ);
    in.print_value(0,NY,NZ);
    in.print_value(NX,NY,NZ);
    return  in(0,0,0)==0. && in(NX,NY,0)==NX+NY && in(NX,0,NZ)==NX+NZ && in(0,NY,NZ)==NY+NZ && in(NX,NY,NZ)==NX+NY+NZ;
#else
    out.print_value(NX,NY,0);
    out.print_value(NX,0,NZ);
    out.print_value(0,NY,NZ);
    out.print_value(NX,NY,NZ);
    return  out(0,0,0)==0. && out(NX,NY,0)==NX+NY && out(NX,0,NZ)==NX+NZ && out(0,NY,NZ)==NY+NZ && out(NX,NY,NZ)==NX+NY+NZ;
#endif

}

}//namespace copy_stencil
