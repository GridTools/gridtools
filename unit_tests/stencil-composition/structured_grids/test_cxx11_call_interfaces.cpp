/*
  GridTools Libraries

  Copyright (c) 2016, GridTools Consortium
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  1. Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.




  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

  For information: http://eth-cscs.github.io/gridtools/
*/
#include "gtest/gtest.h"
#include <stencil-composition/stencil-composition.hpp>
#include <stencil-composition/structured_grids/call_interfaces.hpp>
#include <tools/verifier.hpp>

using namespace gridtools;
using namespace gridtools::enumtype;
using namespace gridtools::expressions;

namespace call_interface_functors {

    typedef interval< level< 0, -2 >, level< 1, 1 > > axis;
    typedef interval< level< 0, -1 >, level< 1, -1 > > x_interval;

    struct copy_functor {
        typedef in_accessor< 0, extent<>, 3 > in;
        typedef inout_accessor< 1, extent<>, 3 > out;
        typedef boost::mpl::vector< in, out > arg_list;
        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation const &eval, x_interval) {
            eval(out()) = eval(in());
        }
    };

    struct call_copy_functor {
        typedef in_accessor< 0, extent<>, 3 > in;
        typedef inout_accessor< 1, extent<>, 3 > out;
        typedef boost::mpl::vector< in, out > arg_list;
        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation const &eval, x_interval) {
            eval(out()) = call< copy_functor, x_interval >::with(eval, in(), out());
        }
    };

    struct call_at_copy_functor {
        typedef in_accessor< 0, extent<>, 3 > in;
        typedef inout_accessor< 1, extent<>, 3 > out;
        typedef boost::mpl::vector< in, out > arg_list;
        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation const &eval, x_interval) {
            eval(out()) = call< copy_functor, x_interval >::at< 1, 1, 0 >::with(eval, in(), out());
        }
    };

    struct call_with_offsets_copy_functor {
        typedef in_accessor< 0, extent<>, 3 > in;
        typedef inout_accessor< 1, extent<>, 3 > out;
        typedef boost::mpl::vector< in, out > arg_list;
        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation const &eval, x_interval) {
            eval(out()) = call< copy_functor, x_interval >::with_offsets(eval, in(1, 1, 0), out());
        }
    };

    struct call_at_with_offsets_copy_functor {
        typedef in_accessor< 0, extent<>, 3 > in;
        typedef inout_accessor< 1, extent<>, 3 > out;
        typedef boost::mpl::vector< in, out > arg_list;
        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation const &eval, x_interval) {
            eval(out()) = call< copy_functor, x_interval >::at< -1, -1, 0 >::with_offsets(eval, in(1, 1, 0), out());
        }
    };

    struct call_call_copy_functor {
        typedef in_accessor< 0, extent<>, 3 > in;
        typedef inout_accessor< 1, extent<>, 3 > out;
        typedef boost::mpl::vector< in, out > arg_list;
        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation const &eval, x_interval) {
            eval(out()) = call< call_copy_functor, x_interval >::with(eval, in(), out());
        }
    };

    struct call_call_at_copy_functor {
        typedef in_accessor< 0, extent<>, 3 > in;
        typedef inout_accessor< 1, extent<>, 3 > out;
        typedef boost::mpl::vector< in, out > arg_list;
        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation const &eval, x_interval) {
            eval(out()) = call< call_at_copy_functor, x_interval >::with(eval, in(), out());
        }
    };

    struct call_call_with_offsets_copy_functor {
        typedef in_accessor< 0, extent<>, 3 > in;
        typedef inout_accessor< 1, extent<>, 3 > out;
        typedef boost::mpl::vector< in, out > arg_list;
        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation const &eval, x_interval) {
            eval(out()) = call< call_with_offsets_copy_functor, x_interval >::with(eval, in(), out());
        }
    };

    struct call_at_call_copy_functor {
        typedef in_accessor< 0, extent<>, 3 > in;
        typedef inout_accessor< 1, extent<>, 3 > out;
        typedef boost::mpl::vector< in, out > arg_list;
        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation const &eval, x_interval) {
            eval(out()) = call< call_copy_functor, x_interval >::at< 1, 1, 0 >::with(eval, in(), out());
        }
    };

    struct call_at_call_at_copy_functor {
        typedef in_accessor< 0, extent<>, 3 > in;
        typedef inout_accessor< 1, extent<>, 3 > out;
        typedef boost::mpl::vector< in, out > arg_list;
        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation const &eval, x_interval) {
            eval(out()) = call< call_at_copy_functor, x_interval >::at< -1, -1, 0 >::with(eval, in(), out());
        }
    };

    struct call_with_offsets_call_at_copy_functor {
        typedef in_accessor< 0, extent<>, 3 > in;
        typedef inout_accessor< 1, extent<>, 3 > out;
        typedef boost::mpl::vector< in, out > arg_list;
        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation const &eval, x_interval) {
            eval(out()) = call< call_at_copy_functor, x_interval >::with_offsets(eval, in(-1, -1, 0), out());
        }
    };

    struct call_at_call_with_offsets_copy_functor {
        typedef in_accessor< 0, extent<>, 3 > in;
        typedef inout_accessor< 1, extent<>, 3 > out;
        typedef boost::mpl::vector< in, out > arg_list;
        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation const &eval, x_interval) {
            eval(out()) = call< call_with_offsets_copy_functor, x_interval >::at< -1, -1, 0 >::with(eval, in(), out());
        }
    };

    struct call_with_offsets_call_with_offsets_copy_functor {
        typedef in_accessor< 0, extent<>, 3 > in;
        typedef inout_accessor< 1, extent<>, 3 > out;
        typedef boost::mpl::vector< in, out > arg_list;
        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation const &eval, x_interval) {
            eval(out()) = call< call_with_offsets_copy_functor, x_interval >::with_offsets(eval, in(-1, -1, 0), out());
        }
    };
}

class call_interface : public testing::Test {
  protected:
#ifdef __CUDACC__
#define BACKEND backend< Cuda, GRIDBACKEND, Block >
#else
#ifdef BACKEND_BLOCK
#define BACKEND backend< Host, GRIDBACKEND, Block >
#else
#define BACKEND backend< Host, GRIDBACKEND, Naive >
#endif
#endif
    const uint_t d1 = 13;
    const uint_t d2 = 9;
    const uint_t d3 = 7;
    const uint_t halo_size = 1;

    typedef gridtools::layout_map< 0, 1, 2 > layout_t;
    typedef gridtools::BACKEND::storage_info< 0, layout_t > meta_t;
    typedef gridtools::BACKEND::storage_type< uint_t, meta_t >::type storage_type;

    meta_t meta_;

    halo_descriptor di;
    halo_descriptor dj;
    gridtools::grid< call_interface_functors::axis > grid;

    verifier verifier_;
    array< array< uint_t, 2 >, 3 > verifier_halos;

    storage_type in;
    storage_type out;
    storage_type reference_unchanged;
    storage_type reference_shifted;

    typedef arg< 0, storage_type > p_in;
    typedef arg< 1, storage_type > p_out;
    typedef boost::mpl::vector< p_in, p_out > accessor_list;

    aggregator_type< accessor_list > domain;

    call_interface()
        : meta_(d1, d2, d3), di(halo_size, halo_size, halo_size, d1 - halo_size - 1, d1),
          dj(halo_size, halo_size, halo_size, d2 - halo_size - 1, d2), grid(di, dj),
#if FLOAT_PRECISION == 4
          verifier_(1e-6),
#else
          verifier_(1e-12),
#endif
          verifier_halos{{{halo_size, halo_size}, {halo_size, halo_size}, {halo_size, halo_size}}}, in(meta_, 0, "in"),
          out(meta_, -5, "out"), reference_unchanged(meta_, -1, "reference_unchanged"),
          reference_shifted(meta_, -1, "reference shifted"), domain(boost::fusion::make_vector(&in, &out)) {
        grid.value_list[0] = 0;
        grid.value_list[1] = d3 - 1;

        in.initialize([](const uint_t &i, const uint_t &j, const uint_t &k) { return i + j * 10 + k * 100; });
        reference_unchanged.initialize(
            [](const uint_t &i, const uint_t &j, const uint_t &k) { return i + j * 10 + k * 100; });
        reference_shifted.initialize(
            [](const uint_t &i, const uint_t &j, const uint_t &k) { return (i + 1) + (j + 1) * 10 + k * 100; });
    }

    template < typename Computation >
    void execute_computation(Computation &comp) {
        comp->ready();
        comp->steady();
        comp->run();
#ifdef __CUDACC__
        out.d2h_update();
#endif
    }
};

TEST_F(call_interface, call_to_copy_functor) {
    auto comp = gridtools::make_computation< gridtools::BACKEND >(
        domain,
        grid,
        gridtools::make_multistage(execute< forward >(),
            gridtools::make_stage< call_interface_functors::call_copy_functor >(p_in(), p_out())));

    execute_computation(comp);

    ASSERT_TRUE(verifier_.verify(grid, reference_unchanged, out, verifier_halos));
}

TEST_F(call_interface, call_at_to_copy_functor) {
    auto comp = gridtools::make_computation< gridtools::BACKEND >(
        domain,
        grid,
        gridtools::make_multistage(execute< forward >(),
            gridtools::make_stage< call_interface_functors::call_at_copy_functor >(p_in(), p_out())));

    execute_computation(comp);

    ASSERT_TRUE(verifier_.verify(grid, reference_shifted, out, verifier_halos));
}

TEST_F(call_interface, call_with_offsets_to_copy_functor) {
    auto comp = gridtools::make_computation< gridtools::BACKEND >(
        domain,
        grid,
        gridtools::make_multistage(execute< forward >(),
            gridtools::make_stage< call_interface_functors::call_with_offsets_copy_functor >(p_in(), p_out())));

    execute_computation(comp);

    ASSERT_TRUE(verifier_.verify(grid, reference_shifted, out, verifier_halos));
}

TEST_F(call_interface, call_at_with_offsets_to_copy_functor) {
    auto comp = gridtools::make_computation< gridtools::BACKEND >(
        domain,
        grid,
        gridtools::make_multistage(execute< forward >(),
            gridtools::make_stage< call_interface_functors::call_at_with_offsets_copy_functor >(p_in(), p_out())));

    execute_computation(comp);

    ASSERT_TRUE(verifier_.verify(grid, reference_unchanged, out, verifier_halos));
}

TEST_F(call_interface, call_to_call_to_copy_functor) {
    auto comp = gridtools::make_computation< gridtools::BACKEND >(
        domain,
        grid,
        gridtools::make_multistage(execute< forward >(),
            gridtools::make_stage< call_interface_functors::call_call_copy_functor >(p_in(), p_out())));

    execute_computation(comp);

    ASSERT_TRUE(verifier_.verify(grid, reference_unchanged, out, verifier_halos));
}

TEST_F(call_interface, call_to_call_at_to_copy_functor) {
    auto comp = gridtools::make_computation< gridtools::BACKEND >(
        domain,
        grid,
        gridtools::make_multistage(execute< forward >(),
            gridtools::make_stage< call_interface_functors::call_call_at_copy_functor >(p_in(), p_out())));

    execute_computation(comp);

    ASSERT_TRUE(verifier_.verify(grid, reference_shifted, out, verifier_halos));
}

TEST_F(call_interface, call_to_call_with_offsets_to_copy_functor) {
    auto comp = gridtools::make_computation< gridtools::BACKEND >(
        domain,
        grid,
        gridtools::make_multistage(execute< forward >(),
            gridtools::make_stage< call_interface_functors::call_call_with_offsets_copy_functor >(p_in(), p_out())));

    execute_computation(comp);

    ASSERT_TRUE(verifier_.verify(grid, reference_shifted, out, verifier_halos));
}

TEST_F(call_interface, call_at_to_call_to_copy_functor) {
    auto comp = gridtools::make_computation< gridtools::BACKEND >(
        domain,
        grid,
        gridtools::make_multistage(execute< forward >(),
            gridtools::make_stage< call_interface_functors::call_at_call_copy_functor >(p_in(), p_out())));

    execute_computation(comp);

    ASSERT_TRUE(verifier_.verify(grid, reference_shifted, out, verifier_halos));
}

TEST_F(call_interface, call_at_to_call_at_to_copy_functor) {
    auto comp = gridtools::make_computation< gridtools::BACKEND >(
        domain,
        grid,
        gridtools::make_multistage(execute< forward >(),
            gridtools::make_stage< call_interface_functors::call_at_call_at_copy_functor >(p_in(), p_out())));

    execute_computation(comp);

    ASSERT_TRUE(verifier_.verify(grid, reference_unchanged, out, verifier_halos));
}

TEST_F(call_interface, call_with_offsets_to_call_at_to_copy_functor) {
    auto comp = gridtools::make_computation< gridtools::BACKEND >(
        domain,
        grid,
        gridtools::make_multistage(execute< forward >(),
            gridtools::make_stage< call_interface_functors::call_with_offsets_call_at_copy_functor >(p_in(), p_out())));

    execute_computation(comp);

    ASSERT_TRUE(verifier_.verify(grid, reference_unchanged, out, verifier_halos));
}

TEST_F(call_interface, call_at_to_call_with_offsets_to_copy_functor) {
    auto comp = gridtools::make_computation< gridtools::BACKEND >(
        domain,
        grid,
        gridtools::make_multistage(execute< forward >(),
            gridtools::make_stage< call_interface_functors::call_at_call_with_offsets_copy_functor >(p_in(), p_out())));

    execute_computation(comp);

    ASSERT_TRUE(verifier_.verify(grid, reference_unchanged, out, verifier_halos));
}

TEST_F(call_interface, call_with_offsets_to_call_with_offsets_to_copy_functor) {
    auto comp = gridtools::make_computation< gridtools::BACKEND >(
        domain,
        grid,
        gridtools::make_multistage(execute< forward >(),
            gridtools::make_stage< call_interface_functors::call_with_offsets_call_with_offsets_copy_functor >(
                                       p_in(), p_out())));

    execute_computation(comp);

    ASSERT_TRUE(verifier_.verify(grid, reference_unchanged, out, verifier_halos));
}

class call_proc_interface : public testing::Test {
  protected:
#ifdef __CUDACC__
#define BACKEND backend< Cuda, GRIDBACKEND, Block >
#else
#ifdef BACKEND_BLOCK
#define BACKEND backend< Host, GRIDBACKEND, Block >
#else
#define BACKEND backend< Host, GRIDBACKEND, Naive >
#endif
#endif
    const uint_t d1 = 13;
    const uint_t d2 = 9;
    const uint_t d3 = 7;
    const uint_t halo_size = 1;

    typedef gridtools::layout_map< 0, 1, 2 > layout_t;
    typedef gridtools::BACKEND::storage_info< 0, layout_t > meta_t;
    typedef gridtools::BACKEND::storage_type< uint_t, meta_t >::type storage_type;

    meta_t meta_;

    halo_descriptor di;
    halo_descriptor dj;
    gridtools::grid< call_interface_functors::axis > grid;

    verifier verifier_;
    array< array< uint_t, 2 >, 3 > verifier_halos;

    storage_type in;
    storage_type out1;
    storage_type out2;
    storage_type reference_unchanged;
    storage_type reference_shifted;

    typedef arg< 0, storage_type > p_in;
    typedef arg< 1, storage_type > p_out1;
    typedef arg< 2, storage_type > p_out2;
    typedef boost::mpl::vector< p_in, p_out1, p_out2 > accessor_list;

    aggregator_type< accessor_list > domain;

    call_proc_interface()
        : meta_(d1, d2, d3), di(halo_size, halo_size, halo_size, d1 - halo_size - 1, d1),
          dj(halo_size, halo_size, halo_size, d2 - halo_size - 1, d2), grid(di, dj),
#if FLOAT_PRECISION == 4
          verifier_(1e-6),
#else
          verifier_(1e-12),
#endif
          verifier_halos{{{halo_size, halo_size}, {halo_size, halo_size}, {halo_size, halo_size}}}, in(meta_, 0, "in"),
          out1(meta_, -5, "out1"), out2(meta_, -5, "out2"), reference_unchanged(meta_, -1, "reference_unchanged"),
          reference_shifted(meta_, -1, "reference shifted"), domain(boost::fusion::make_vector(&in, &out1, &out2)) {
        grid.value_list[0] = 0;
        grid.value_list[1] = d3 - 1;

        in.initialize([](const uint_t &i, const uint_t &j, const uint_t &k) { return i + j * 10 + k * 100; });
        reference_unchanged.initialize(
            [](const uint_t &i, const uint_t &j, const uint_t &k) { return i + j * 10 + k * 100; });
        reference_shifted.initialize(
            [](const uint_t &i, const uint_t &j, const uint_t &k) { return (i + 1) + (j + 1) * 10 + k * 100; });
    }

    template < typename Computation >
    void execute_computation(Computation &comp) {
        comp->ready();
        comp->steady();
        comp->run();
#ifdef __CUDACC__
        out1.d2h_update();
        out2.d2h_update();
#endif
    }
};

namespace call_proc_interface_functors {
    typedef interval< level< 0, -2 >, level< 1, 1 > > axis;
    typedef interval< level< 0, -1 >, level< 1, -1 > > x_interval;

    struct copy_functor {
        typedef in_accessor< 0, extent<>, 3 > in;
        typedef inout_accessor< 1, extent<>, 3 > out;
        typedef boost::mpl::vector< in, out > arg_list;
        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation const &eval, x_interval) {
            eval(out()) = eval(in());
        }
    };

    struct copy_twice_functor {
        typedef in_accessor< 0, extent<>, 3 > in;
        typedef inout_accessor< 1, extent<>, 3 > out1;
        typedef inout_accessor< 2, extent<>, 3 > out2;
        typedef boost::mpl::vector< in, out1, out2 > arg_list;
        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation const &eval, x_interval) {
            eval(out1()) = eval(in());
            eval(out2()) = eval(in());
        }
    };

    struct call_copy_twice_functor {
        typedef in_accessor< 0, extent<>, 3 > in;
        typedef inout_accessor< 1, extent<>, 3 > out1;
        typedef inout_accessor< 2, extent<>, 3 > out2;
        typedef boost::mpl::vector< in, out1, out2 > arg_list;
        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation const &eval, x_interval) {
            call_proc< copy_twice_functor, x_interval >::with(eval, in(), out1(), out2());
        }
    };

    struct call_with_offsets_copy_twice_functor {
        typedef in_accessor< 0, extent<>, 3 > in;
        typedef inout_accessor< 1, extent<>, 3 > out1;
        typedef inout_accessor< 2, extent<>, 3 > out2;
        typedef boost::mpl::vector< in, out1, out2 > arg_list;
        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation const &eval, x_interval) {
            call_proc< copy_twice_functor, x_interval >::with_offsets(eval, in(1, 1, 0), out1(), out2());
        }
    };

    struct call_at_with_offsets_copy_twice_functor {
        typedef in_accessor< 0, extent<>, 3 > in;
        typedef inout_accessor< 1, extent<>, 3 > out1;
        typedef inout_accessor< 2, extent<>, 3 > out2;
        typedef boost::mpl::vector< in, out1, out2 > arg_list;
        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation const &eval, x_interval) {
            call_proc< copy_twice_functor, x_interval >::at< 1, 1, 0 >::with_offsets(
                eval, in(), out1(-1, -1, 0), out2(-1, -1, 0)); // outs are at the original position
        }
    };

    struct call_call_copy_twice_functor {
        typedef in_accessor< 0, extent<>, 3 > in;
        typedef inout_accessor< 1, extent<>, 3 > out1;
        typedef inout_accessor< 2, extent<>, 3 > out2;
        typedef boost::mpl::vector< in, out1, out2 > arg_list;
        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation const &eval, x_interval) {
            call_proc< call_copy_twice_functor, x_interval >::with(eval, in(), out1(), out2());
        }
    };

    struct call_with_offsets_call_copy_twice_functor {
        typedef in_accessor< 0, extent<>, 3 > in;
        typedef inout_accessor< 1, extent<>, 3 > out1;
        typedef inout_accessor< 2, extent<>, 3 > out2;
        typedef boost::mpl::vector< in, out1, out2 > arg_list;
        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation const &eval, x_interval) {
            call_proc< call_copy_twice_functor, x_interval >::with_offsets(eval, in(1, 1, 0), out1(), out2());
        }
    };

    struct call_with_offsets_call_with_offsets_copy_twice_functor {
        typedef in_accessor< 0, extent<>, 3 > in;
        typedef inout_accessor< 1, extent<>, 3 > out1;
        typedef inout_accessor< 2, extent<>, 3 > out2;
        typedef boost::mpl::vector< in, out1, out2 > arg_list;
        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation const &eval, x_interval) {
            call_proc< call_with_offsets_copy_twice_functor, x_interval >::with_offsets(
                eval, in(-1, -1, 0), out1(), out2());
        }
    };

    struct call_with_local_variable {
        typedef in_accessor< 0, extent<>, 3 > in;
        typedef inout_accessor< 1, extent<>, 3 > out1;
        typedef inout_accessor< 2, extent<>, 3 > out2;
        typedef boost::mpl::vector< in, out1, out2 > arg_list;
        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation const &eval, x_interval) {
            double local_in = 1;
            double local_out = -1;

            call_proc< copy_functor, x_interval >::with(eval, local_in, local_out);

            if (local_out > 0.) {
                eval(out1()) = eval(in());
            }
        }
    };
}

TEST_F(call_proc_interface, call_to_copy_twice_functor) {
    auto comp = gridtools::make_computation< gridtools::BACKEND >(
        domain,
        grid,
        gridtools::make_multistage(execute< forward >(),
            gridtools::make_stage< call_proc_interface_functors::call_copy_twice_functor >(
                                       p_in(), p_out1(), p_out2())));

    execute_computation(comp);

    ASSERT_TRUE(verifier_.verify(grid, reference_unchanged, out1, verifier_halos));
    ASSERT_TRUE(verifier_.verify(grid, reference_unchanged, out2, verifier_halos));
}

TEST_F(call_proc_interface, call_with_offsets_to_copy_twice_functor) {
    auto comp = gridtools::make_computation< gridtools::BACKEND >(
        domain,
        grid,
        gridtools::make_multistage(execute< forward >(),
            gridtools::make_stage< call_proc_interface_functors::call_with_offsets_copy_twice_functor >(
                                       p_in(), p_out1(), p_out2())));

    execute_computation(comp);

    ASSERT_TRUE(verifier_.verify(grid, reference_shifted, out1, verifier_halos));
    ASSERT_TRUE(verifier_.verify(grid, reference_shifted, out2, verifier_halos));
}

TEST_F(call_proc_interface, call_at_with_offsets_to_copy_twice_functor) {
    auto comp = gridtools::make_computation< gridtools::BACKEND >(
        domain,
        grid,
        gridtools::make_multistage(execute< forward >(),
            gridtools::make_stage< call_proc_interface_functors::call_at_with_offsets_copy_twice_functor >(
                                       p_in(), p_out1(), p_out2())));

    execute_computation(comp);

    ASSERT_TRUE(verifier_.verify(grid, reference_shifted, out1, verifier_halos));
    ASSERT_TRUE(verifier_.verify(grid, reference_shifted, out2, verifier_halos));
}

TEST_F(call_proc_interface, call_to_call_to_copy_twice_functor) {
    auto comp = gridtools::make_computation< gridtools::BACKEND >(
        domain,
        grid,
        gridtools::make_multistage(execute< forward >(),
            gridtools::make_stage< call_proc_interface_functors::call_call_copy_twice_functor >(
                                       p_in(), p_out1(), p_out2())));

    execute_computation(comp);

    ASSERT_TRUE(verifier_.verify(grid, reference_unchanged, out1, verifier_halos));
    ASSERT_TRUE(verifier_.verify(grid, reference_unchanged, out2, verifier_halos));
}

TEST_F(call_proc_interface, call_with_offsets_to_call_to_copy_twice_functor) {
    auto comp = gridtools::make_computation< gridtools::BACKEND >(
        domain,
        grid,
        gridtools::make_multistage(execute< forward >(),
            gridtools::make_stage< call_proc_interface_functors::call_with_offsets_call_copy_twice_functor >(
                                       p_in(), p_out1(), p_out2())));

    execute_computation(comp);

    ASSERT_TRUE(verifier_.verify(grid, reference_shifted, out1, verifier_halos));
    ASSERT_TRUE(verifier_.verify(grid, reference_shifted, out2, verifier_halos));
}

TEST_F(call_proc_interface, call_with_offsets_to_call_with_offsets_to_copy_twice_functor) {
    auto comp = gridtools::make_computation< gridtools::BACKEND >(
        domain,
        grid,
        gridtools::make_multistage(
            execute< forward >(),
            gridtools::make_stage<
                call_proc_interface_functors::call_with_offsets_call_with_offsets_copy_twice_functor >(
                p_in(), p_out1(), p_out2())));

    execute_computation(comp);

    ASSERT_TRUE(verifier_.verify(grid, reference_unchanged, out1, verifier_halos));
    ASSERT_TRUE(verifier_.verify(grid, reference_unchanged, out2, verifier_halos));
}

TEST_F(call_proc_interface, call_using_local_variables) {
    auto comp = gridtools::make_computation< gridtools::BACKEND >(
        domain,
        grid,
        gridtools::make_multistage(execute< forward >(),
            gridtools::make_stage< call_proc_interface_functors::call_with_local_variable >(
                                       p_in(), p_out1(), p_out2())));

    execute_computation(comp);

    ASSERT_TRUE(verifier_.verify(grid, reference_unchanged, out1, verifier_halos));
}
