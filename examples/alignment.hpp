#pragma once

#include <stencil-composition/stencil-composition.hpp>

/**
  @file
  This file shows an implementation of the "copy" stencil, simple copy of one field done on the backend,
  in which a misaligned storage is aligned
*/

using gridtools::level;
using gridtools::accessor;
using gridtools::extent;
using gridtools::arg;

using namespace gridtools;
using namespace enumtype;

namespace aligned_copy_stencil {
#ifdef __CUDACC__
    typedef gridtools::layout_map< 2, 1, 0 > layout_t; // stride 1 on i
#else
    typedef gridtools::layout_map< 0, 1, 2 > layout_t; // stride 1 on k
#endif

// random padding
#ifdef __CUDACC__
    typedef halo< 2, 0, 0 > halo_t;
    typedef aligned< 32 > alignment_t;
#else
    typedef aligned< 32 > alignment_t;
#ifdef CXX11_ENABLED
    typedef halo< 0, 0, 2 > halo_t;
#else
    typedef halo< 0, 2, 0 > halo_t;
#endif
#endif
    // This is the definition of the special regions in the "vertical" direction
    typedef gridtools::interval< level< 0, -1 >, level< 1, -1 > > x_interval;
    typedef gridtools::interval< level< 0, -2 >, level< 1, 1 > > axis;

    // These are the stencil operators that compose the multistage stencil in this test
    struct copy_functor {

#ifdef __CUDACC__
        /** @brief checking all storages alignment using a specific storage_info

            \param storage_id ordinal number identifying the storage_info checked
            \param boundary ordinal number identifying the alignment
        */
        template < typename ItDomain >
        GT_FUNCTION static bool check_pointer_alignment(ItDomain const &it_domain, uint_t storage_id, uint_t boundary) {
            bool result_ = true;
            if (threadIdx.x == 0) {
                for (ushort_t i = 0; i < ItDomain::iterate_domain_t::N_DATA_POINTERS; ++i) {
                    result_ = (bool)(result_ && (bool)(((size_t)(it_domain.get().data_pointer()[i] +
                                                                 it_domain.get().index()[storage_id]) &
                                                           (boundary - 1)) == 0));
                    if (!result_) {
                        printf("[storage # %d,", i);
                        printf("index %d]", it_domain.get().index()[storage_id]);
                        printf(" pointer: %x ",
                            (size_t)it_domain.get().data_pointer()[i] + it_domain.get().index()[storage_id]);
                        break;
                    }
                }
            }
            return result_;
        }
#endif

        typedef accessor< 0, enumtype::in, extent< 0, 0, 0, 0 >, 3 > in;
        typedef accessor< 1, enumtype::inout, extent< 0, 0, 0, 0 >, 3 > out;
        typedef boost::mpl::vector< in, out > arg_list;

        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation const &eval, x_interval) {

#ifdef __CUDACC__
#ifndef NDEBUG
            if (!check_pointer_alignment(eval, 0, alignment_t::value)) {
                printf("alignment error in some storages with first meta_storage \n");
            }
#endif
#endif
            eval(out()) = eval(in());
        }
    };

#ifdef __CUDACC__
#define BACKEND backend< Cuda, GRIDBACKEND, Block >
#else
#ifdef BACKEND_BLOCK
#define BACKEND backend< Host, GRIDBACKEND, Block >
#else
#define BACKEND backend< Host, GRIDBACKEND, Naive >
#endif
#endif

    typedef gridtools::BACKEND::storage_info< 0, layout_t, halo_t, alignment_t > meta_data_t;

    bool test(uint_t d1, uint_t d2, uint_t d3) {

        meta_data_t meta_data_(d1, d2, d3);

        //                   strides  1 x xy
        //                      dims  x y z
        typedef gridtools::BACKEND::storage_type< float_type, meta_data_t >::type storage_t;

        // Definition of the actual data fields that are used for input/output
        typedef storage_t storage_type;
        storage_type in(meta_data_, "in");
        storage_type out(meta_data_, (float_type)-1.);
        for (uint_t i = halo_t::get< 0 >(); i < d1 + halo_t::get< 0 >(); ++i)
            for (uint_t j = halo_t::get< 1 >(); j < d2 + halo_t::get< 1 >(); ++j)
                for (uint_t k = halo_t::get< 2 >(); k < d3 + halo_t::get< 2 >(); ++k) {
                    in(i, j, k) = i + j + k;
                }

        typedef arg< 0, storage_type > p_in;
        typedef arg< 1, storage_type > p_out;

        typedef boost::mpl::vector< p_in, p_out > accessor_list;
        // construction of the domain. The domain is the physical domain of the problem, with all the physical fields
        // that are used, temporary and not
        // It must be noted that the only fields to be passed to the constructor are the non-temporary.
        // The order in which they have to be passed is the order in which they appear scanning the placeholders in
        // order. (I don't particularly like this)
        gridtools::domain_type< accessor_list > domain(boost::fusion::make_vector(&in, &out));

        // Definition of the physical dimensions of the problem.
        // The constructor takes the horizontal plane dimensions,
        // while the vertical ones are set according the the axis property soon after
        // gridtools::coordinates<axis> grid(2,d1-2,2,d2-2);
        uint_t di[5] = {
            halo_t::get< 0 >(), 0, halo_t::get< 0 >(), d1 + halo_t::get< 0 >() - 1, d1 + halo_t::get< 0 >()};
        uint_t dj[5] = {
            halo_t::get< 1 >(), 0, halo_t::get< 1 >(), d2 + halo_t::get< 1 >() - 1, d2 + halo_t::get< 1 >()};

        gridtools::grid< axis > grid(di, dj);

        grid.value_list[0] = halo_t::get< 2 >();
        grid.value_list[1] = d3 + halo_t::get< 2 >() - 1;

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
        gridtools::stencil *
#else
        boost::shared_ptr< gridtools::stencil >
#endif
#endif
            copy = gridtools::make_computation< gridtools::BACKEND >(
                domain,
                grid,
                gridtools::make_mss // mss_descriptor
                (execute< forward >(),
                    gridtools::make_esf< copy_functor >(p_in() // esf_descriptor
                        ,
                        p_out())));

        copy->ready();

        copy->steady();

        copy->run();

        copy->finalize();

#ifdef BENCHMARK
        std::cout << copy->print_meter() << std::endl;
#endif

        bool success = true;
        for (uint_t i = halo_t::get< 0 >(); i < d1 + halo_t::get< 0 >(); ++i)
            for (uint_t j = halo_t::get< 1 >(); j < d2 + halo_t::get< 1 >(); ++j)
                for (uint_t k = halo_t::get< 2 >(); k < d3 + halo_t::get< 2 >(); ++k) {
                    if (in(i, j, k) != out(i, j, k) || out(i, j, k) != i + j + k) {
                        std::cout << "error in " << i << ", " << j << ", " << k << ": "
                                  << "in = " << in(i, j, k) << ", out = " << out(i, j, k) << std::endl;
                        success = false;
                    }
                }
        return success;
    }
} // namespace aligned_copy_stencil
