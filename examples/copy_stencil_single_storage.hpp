#pragma once

#include <stencil-composition/stencil-composition.hpp>

/**
  @file
  This file shows an implementation of the "copy" stencil, simple copy of one field done on the backend
*/

using gridtools::level;
using gridtools::accessor;
using gridtools::extent;
using gridtools::arg;

using namespace gridtools;
using namespace enumtype;

namespace copy_stencil {
#ifdef __CUDACC__
    typedef gridtools::layout_map< 2, 1, 0 > layout_t; // stride 1 on i
#else
    typedef gridtools::layout_map< 0, 1, 2 > layout_t; // stride 1 on k
#endif

    // This is the definition of the special regions in the "vertical" direction
    typedef gridtools::interval< level< 0, -1 >, level< 1, -1 > > x_interval;
    typedef gridtools::interval< level< 0, -2 >, level< 1, 1 > > axis;

    // These are the stencil operators that compose the multistage stencil in this test
    struct copy_functor {

        typedef accessor< 0, enumtype::inout, extent< 0, 0, 0, 0 >, 4 > in;
        typedef boost::mpl::vector< in > arg_list;

        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation const &eval, x_interval) {
            eval(in()) = eval(in(dimension< 4 >(1)));
        }
    };

    /*
     * The following operators and structs are for debugging only
     */
    std::ostream &operator<<(std::ostream &s, copy_functor const) { return s << "copy_functor"; }

    void handle_error(int_t) { std::cout << "error" << std::endl; }

    bool test(uint_t x, uint_t y, uint_t z) {

        uint_t d1 = x;
        uint_t d2 = y;
        uint_t d3 = z;

#ifdef __CUDACC__
#define BACKEND backend< Cuda, GRIDBACKEND, Block >
#else
#ifdef BACKEND_BLOCK
#define BACKEND backend< Host, GRIDBACKEND, Block >
#else
#define BACKEND backend< Host, GRIDBACKEND, Naive >
#endif
#endif

        typedef BACKEND::storage_info< 0, layout_t > meta_data_t;
        meta_data_t meta_data_(x, y, z);

        //                   strides  1 x xy
        //                      dims  x y z
        typedef gridtools::BACKEND::storage_type< float_type, meta_data_t >::type storage_t;

        // Definition of the actual data fields that are used for input/output
        typedef field< storage_t, 2 >::type storage_type;
        storage_type in(meta_data_);
        in.initialize(0.);
        for (uint_t i = 0; i < d1; ++i) {
            for (uint_t j = 0; j < d2; ++j) {
                for (uint_t k = 0; k < d3; ++k) {
                    in.template get_value< 1, 0 >(i, j, k) = i + j + k;
                }
            }
        }

        typedef arg< 0, storage_type > p_in;

        typedef boost::mpl::vector< p_in > accessor_list;
        // construction of the domain. The domain is the physical domain of the problem, with all the physical fields
        // that are used, temporary and not
        // It must be noted that the only fields to be passed to the constructor are the non-temporary.
        // The order in which they have to be passed is the order in which they appear scanning the placeholders in
        // order. (I don't particularly like this)
        gridtools::domain_type< accessor_list > domain(boost::fusion::make_vector(&in));

        // Definition of the physical dimensions of the problem.
        // The constructor takes the horizontal plane dimensions,
        // while the vertical ones are set according the the axis property soon after
        // gridtools::grid<axis> grid(2,d1-2,2,d2-2);
        uint_t di[5] = {0, 0, 0, d1 - 1, d1};
        uint_t dj[5] = {0, 0, 0, d2 - 1, d2};

        gridtools::grid< axis > grid(di, dj);
        grid.value_list[0] = 0;
        grid.value_list[1] = d3 - 1;

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
#ifdef CXX11_ENABLED
        auto
#else
#ifdef __CUDACC__
        gridtools::computation *
#else
        boost::shared_ptr< gridtools::computation >
#endif
#endif
            copy = gridtools::make_computation< gridtools::BACKEND >(
                domain,
                grid,
                gridtools::make_mss // mss_descriptor
                (execute< forward >(),
                    gridtools::make_esf< copy_functor >(p_in() // esf_descriptor
                        )));

        copy->ready();

        copy->steady();

        copy->run();

        copy->finalize();

#ifdef BENCHMARK
        std::cout << copy->print_meter() << std::endl;
#endif

        bool success = true;
        for (uint_t i = 0; i < d1; ++i) {
            for (uint_t j = 0; j < d2; ++j) {
                for (uint_t k = 0; k < d3; ++k) {
                    if (in.get_value< 0, 0 >(i, j, k) != in.get_value< 1, 0 >(i, j, k)) {
                        std::cout << "error in " << i << ", " << j << ", " << k << ": "
                                  << "in = " << (in.get_value< 0, 0 >(i, j, k))
                                  << ", out = " << (in.get_value< 1, 0 >(i, j, k)) << std::endl;
                        success = false;
                    }
                }
            }
        }
        return success;
    }
} // namespace copy_stencil
