#pragma once


#include <gridtools.hpp>
#include <stencil-composition/backend.hpp>
#include <stencil-composition/make_computation.hpp>
#include <stencil-composition/interval.hpp>


#ifdef USE_PAPI_WRAP
#include <papi_wrap.hpp>
#include <papi.hpp>
#endif

/**
  @file
  This file shows an implementation of the "copy" stencil, simple copy of one field done on the backend
*/

using gridtools::level;
using gridtools::accessor;
using gridtools::range;
using gridtools::arg;

using namespace gridtools;
using namespace enumtype;


namespace copy_stencil{
#ifdef __CUDACC__
        typedef gridtools::layout_map<2,1,0> layout_t;//stride 1 on i
#else
        typedef gridtools::layout_map<0,1,2> layout_t;//stride 1 on k
#endif

    // This is the definition of the special regions in the "vertical" direction
    typedef gridtools::interval<level<0,-1>, level<1,-1> > x_interval;
    typedef gridtools::interval<level<0,-2>, level<1,1> > axis;

    // These are the stencil operators that compose the multistage stencil in this test
    struct copy_functor {

        typedef const accessor<0, range<0,0,0,0>, 3> in;
        typedef accessor<1, range<0,0,0,0>, 3> out;
        typedef boost::mpl::vector<in,out> arg_list;

        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_interval) {
            eval(out())=eval(in());
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

    typedef meta_storage< 0,layout_t, false > meta_data_t;

    // typedef meta_storage_wrapper<meta_storage<0,layout_t, false> > meta_data_t;
    bool test(uint_t x, uint_t y, uint_t z) {

        meta_data_t meta_data_(x,y,z);

#ifdef USE_PAPI_WRAP
        int collector_init = pw_new_collector("Init");
        int collector_execute = pw_new_collector("Execute");
#endif

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

        //                   strides  1 x xy
        //                      dims  x y z
        typedef gridtools::BACKEND::storage_type<float_type, meta_data_t >::type storage_type;

        // Definition of the actual data fields that are used for input/output
        storage_type in(meta_data_, "in");
        storage_type out(meta_data_, -1.);

        for(uint_t i=0; i<d1; ++i)
            for(uint_t j=0; j<d2; ++j)
                for(uint_t k=0; k<d3; ++k)
                {
                    in(i, j, k)=i+j+k;
                }

        typedef arg<0, storage_type > p_in;
        typedef arg<1, storage_type > p_out;

        typedef boost::mpl::vector<p_in, p_out> accessor_list;
        // construction of the domain. The domain is the physical domain of the problem, with all the physical fields that are used, temporary and not
        // It must be noted that the only fields to be passed to the constructor are the non-temporary.
        // The order in which they have to be passed is the order in which they appear scanning the placeholders in order. (I don't particularly like this)
        gridtools::domain_type<accessor_list> domain
            (boost::fusion::make_vector(&in, &out));

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
            gridtools::make_computation<gridtools::BACKEND>
            (
                gridtools::make_mss // mss_descriptor
                (
                    execute<forward>(),
                    gridtools::make_esf<copy_functor>(
                        p_in() // esf_descriptor
                       ,p_out()
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
        copy->finalize();

#ifdef BENCHMARK
        std::cout << copy->print_meter() << std::endl;
#endif

#ifdef USE_PAPI_WRAP
        pw_print();
#endif
        bool success = true;
//         for(uint_t i=0; i<d1; ++i)
//             for(uint_t j=0; j<d2; ++j)
//                 for(uint_t k=0; k<d3; ++k)
//                 {
// #ifdef CXX11_ENABLED
//                     if (in.get_value<0,0>(i, j, k)!=in.get_value<1,0>(i,j,k))
// #else
//                         if (in(i, j, k)!=out(i,j,k))
// #endif
//                         {
//                             std::cout << "error in "
//                                       << i << ", "
//                                       << j << ", "
//                                       << k << ": "
// #ifdef CXX11_ENABLED
//                                       << "in = " << (in.get_value<0,0>(i, j, k))
//                                       << ", out = " << (in.get_value<1,0>(i, j, k))
// #else
//                                       << "in = " << in(i, j, k)
//                                       << ", out = " << out(i, j, k)
// #endif
//                                       << std::endl;
//                             success = false;
//                         }
//                 }
//         if(!success) std::cout << "ERROR" << std::endl;
//         else std::cout << "OK" << std::endl;

        return success;
    }
}//namespace copy_stencil
