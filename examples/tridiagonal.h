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
  @file This file shows an implementation of the Thomas algorithm, done using stencil operations.

  Important convention: the linear system as usual is represented with 4 vectors: the main diagonal
  (diag), the upper and lower first diagonals (sup and inf respectively), and the right hand side
  (rhs). Note that the dimensions and the memory layout are, for an NxN system
  rank(diag)=N       [xxxxxxxxxxxxxxxxxxxxxxxx]
  rank(inf)=N-1      [0xxxxxxxxxxxxxxxxxxxxxxx]
  rank(sup)=N-1      [xxxxxxxxxxxxxxxxxxxxxxx0]
  rank(rhs)=N        [xxxxxxxxxxxxxxxxxxxxxxxx]
  where x danote any numberm and 0 denotes the padding, a dummy value which is not used in
  the algorithm. This choice coresponds to having the same vector index for each row of the matrix.
 */

using gridtools::level;
using gridtools::arg_type;
using gridtools::range;
using gridtools::arg;

using namespace gridtools;
using namespace enumtype;

using namespace expressions;

// This is the definition of the special regions in the "vertical" direction
typedef gridtools::interval<level<0,-1>, level<1,-2> > x_internal;
typedef gridtools::interval<level<0,-2>, level<0,-2> > x_first;
typedef gridtools::interval<level<1,-1>, level<1,-1> > x_last;
typedef gridtools::interval<level<0,-2>, level<1,3> > axis;

struct forward_thomas{
    static const int n_args = 5;
//for vectors: output, and the 3 diagonals

    typedef arg_type<0> out;
    typedef arg_type<1> inf; //a
    typedef arg_type<2> diag; //b
    typedef arg_type<3> sup; //c
    typedef arg_type<4> rhs; //d
    typedef boost::mpl::vector<out, inf, diag, sup, rhs> arg_list;

    template <typename Domain>
    GT_FUNCTION
    static inline void shared_kernel(Domain const& dom) {
        dom(sup()) = dom(sup())/(dom(diag())-dom(sup(z(-1)))*dom(inf()));
        dom(rhs()) = (dom(rhs())-dom(inf())*dom(rhs(z(-1))))/(dom(diag())-dom(sup(z(-1)))*dom(inf()));
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
        dom(sup()) = dom(sup()/diag());
        dom(rhs()) = dom(rhs()/diag());
    }

};

struct backward_thomas{
    static const int n_args = 5;

    typedef arg_type<0> out;
    typedef arg_type<1> inf; //a
    typedef arg_type<2> diag; //b
    typedef arg_type<3> sup; //c
    typedef arg_type<4> rhs; //d
    typedef boost::mpl::vector<out, inf, diag, sup, rhs> arg_list;

    template <typename Domain>
    GT_FUNCTION
    static void shared_kernel(Domain& dom) {
        dom(out()) = dom(rhs())-dom(sup())*dom(out(z(1)));
        //dom(out()) = (dom(rhs())-dom(sup())*dom(out(z(1))))/dom(diag());
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


bool tridiagonal(int x, int y, int z) {

#ifdef USE_PAPI_WRAP
  int collector_init = pw_new_collector("Init");
  int collector_execute = pw_new_collector("Execute");
#endif

    int d1 = x;
    int d2 = y;
    int d3 = z;

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

    typedef gridtools::BACKEND::storage_type<double, gridtools::layout_map<0,1,2> >::type storage_type;
    typedef gridtools::BACKEND::temporary_storage_type<double, gridtools::layout_map<0,1,2> >::type tmp_storage_type;

     // Definition of the actual data fields that are used for input/output
    //storage_type in(d1,d2,d3,-1, std::string("in"));
    storage_type out(d1,d2,d3,0., std::string("out"));
    storage_type inf(d1,d2,d3,-1., std::string("inf"));
    storage_type diag(d1,d2,d3,3., std::string("diag"));
    storage_type sup(d1,d2,d3,1., std::string("sup"));
    storage_type rhs(d1,d2,d3,3., std::string("rhs"));
    for(int i=0; i<d1; ++i)
        for(int j=0; j<d2; ++j)
        {
            rhs(i, j, 2)=4.;
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
    int di[5] = {2, 2, 2, d1-2, d1};
    int dj[5] = {2, 2, 2, d2-2, d2};

    gridtools::coordinates<axis> coords(di, dj);
    coords.value_list[0] = 3;
    coords.value_list[1] = d3-3;

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
    gridtools::computation* forward_step =
#else
        boost::shared_ptr<gridtools::computation> forward_step =
#endif
        gridtools::make_computation<gridtools::BACKEND>
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
        gridtools::make_computation<gridtools::BACKEND>
        (
            gridtools::make_mss // mss_descriptor
            (
                execute<backward>(),
                gridtools::make_esf<backward_thomas>(p_out(), p_inf(), p_diag(), p_sup(), p_rhs()) // esf_descriptor
                ),
            domain, coords
            );

    forward_step->ready();
    forward_step->steady();
    domain.clone_to_gpu();

#ifdef USE_PAPI_WRAP
    pw_stop_collector(collector_init);
#endif

    boost::timer::cpu_timer time;
#ifdef USE_PAPI
if( PAPI_start(event_set) != PAPI_OK)
    handle_error(1);
#endif
#ifdef USE_PAPI_WRAP
    pw_start_collector(collector_execute);
#endif
    forward_step->run();

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
    boost::timer::cpu_times lapse_time = time.elapsed();

    forward_step->finalize();

    // printf("Print OUT field (forward)\n");
    // out.print();
    // printf("Print SUP field (forward)\n");
    // sup.print();
    // printf("Print RHS field (forward)\n");
    // rhs.print();

    backward_step->ready();
    backward_step->steady();
    domain.clone_to_gpu();

#ifdef USE_PAPI_WRAP
    pw_stop_collector(collector_init);
#endif

#ifdef USE_PAPI
if( PAPI_start(event_set) != PAPI_OK)
    handle_error(1);
#endif
#ifdef USE_PAPI_WRAP
    pw_start_collector(collector_execute);
#endif
    backward_step->run();

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

    backward_step->finalize();

#ifdef CUDA_EXAMPLE
    out.m_data.update_cpu();
#endif

    //    in.print();
    printf("Print OUT field\n");
    out.print();
    printf("Print SUP field\n");
    sup.print();
    printf("Print RHS field\n");
    rhs.print();
    //    lap.print();

    std::cout << "TIME " << boost::timer::format(lapse_time) << std::endl;

#ifdef USE_PAPI_WRAP
    pw_print();
#endif

    return // lapse_time.wall<5000000 &&
// #ifdef USE_PAPI
//                     values[0]>1000 && //random value
// #endif
                                true;
}
