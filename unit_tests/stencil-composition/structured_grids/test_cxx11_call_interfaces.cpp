/*
  GridTools Libraries

  Copyright (c) 2017, ETH Zurich and MeteoSwiss
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
#include <stencil-composition/stencil-functions/stencil-functions.hpp>
#include <tools/verifier.hpp>

using namespace gridtools;
using namespace gridtools::enumtype;
using namespace gridtools::expressions;

namespace call_interface_functors {

    using axis = interval< level< 0, -1 >, level< 1, 1 > >;
    using x_interval = interval< level< 0, -1 >, level< 1, -1 > >;
    using smaller_interval = interval< level< 0, 1 >, level< 1, -2 > >;

    struct copy_functor {
        typedef in_accessor< 0, extent<>, 3 > in;
        typedef inout_accessor< 1, extent<>, 3 > out;
        typedef boost::mpl::vector< in, out > arg_list;
        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation &eval, x_interval) {
            eval(out()) = eval(in());
        }
    };

    // The implementation is different for this case, so we should test it
    struct copy_functor_with_out_first {
        typedef inout_accessor< 0, extent<>, 3 > out;
        typedef in_accessor< 1, extent<>, 3 > in;
        typedef boost::mpl::vector< out, in > arg_list;
        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation &eval, x_interval) {
            eval(out()) = eval(in());
        }
    };

    struct call_copy_functor {
        typedef in_accessor< 0, extent<>, 3 > in;
        typedef inout_accessor< 1, extent<>, 3 > out;
        typedef boost::mpl::vector< in, out > arg_list;
        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation &eval, x_interval) {
            eval(out()) = call< copy_functor, x_interval >::with(eval, in());
        }
    };

    struct call_copy_functor_with_out_first {
        typedef in_accessor< 0, extent<>, 3 > in;
        typedef inout_accessor< 1, extent<>, 3 > out;
        typedef boost::mpl::vector< in, out > arg_list;
        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation &eval, x_interval) {
            eval(out()) = call< copy_functor_with_out_first, x_interval >::with(eval, in());
        }
    };

    struct copy_functor_with_expression {
        typedef in_accessor< 0, extent<>, 3 > in;
        typedef inout_accessor< 1, extent<>, 3 > out;
        typedef boost::mpl::vector< in, out > arg_list;
        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation &eval, x_interval) {
            eval(out()) = eval(in() + 0.);
        }
    };

    struct call_copy_functor_with_expression {
        typedef in_accessor< 0, extent<>, 3 > in;
        typedef inout_accessor< 1, extent<>, 3 > out;
        typedef boost::mpl::vector< in, out > arg_list;
        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation &eval, x_interval) {
            eval(out()) = call< copy_functor_with_expression, x_interval >::with(eval, in());
        }
    };

    struct call_at_copy_functor {
        typedef in_accessor< 0, extent<>, 3 > in;
        typedef inout_accessor< 1, extent<>, 3 > out;
        typedef boost::mpl::vector< in, out > arg_list;
        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation &eval, x_interval) {
            eval(out()) = call< copy_functor, x_interval >::at< 1, 1, 0 >::with(eval, in());
        }
    };

    struct call_with_offsets_copy_functor {
        typedef in_accessor< 0, extent<>, 3 > in;
        typedef inout_accessor< 1, extent<>, 3 > out;
        typedef boost::mpl::vector< in, out > arg_list;
        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation &eval, x_interval) {
            eval(out()) = call< copy_functor, x_interval >::with(eval, in(1, 1, 0));
        }
    };

    struct copy_functor_default_interval {
        typedef in_accessor< 0, extent<>, 3 > in;
        typedef inout_accessor< 1, extent<>, 3 > out;
        typedef boost::mpl::vector< in, out > arg_list;
        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation &eval) {
            eval(out()) = eval(in());
        }
    };

    struct call_copy_functor_default_interval {
        typedef in_accessor< 0, extent<>, 3 > in;
        typedef inout_accessor< 1, extent<>, 3 > out;
        typedef boost::mpl::vector< in, out > arg_list;
        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation &eval) {
            eval(out()) = call< copy_functor_default_interval >::with(eval, in());
        }
    };

    struct call_copy_functor_default_interval_from_smaller_interval {
        typedef in_accessor< 0, extent<>, 3 > in;
        typedef inout_accessor< 1, extent<>, 3 > out;
        typedef boost::mpl::vector< in, out > arg_list;
        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation &eval, smaller_interval) {
            eval(out()) = call< copy_functor_default_interval >::with(eval, in());
        }
    };

    struct call_copy_functor_default_interval_with_offset_in_k {
        typedef in_accessor< 0, extent<>, 3 > in;
        typedef inout_accessor< 1, extent<>, 3 > out;
        typedef boost::mpl::vector< in, out > arg_list;
        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation &eval) {
            eval(out()) = call< copy_functor_default_interval >::at< 0, 0, -1 >::with(eval, in(0, 0, 1));
        }
    };

    struct call_at_with_offsets_copy_functor {
        typedef in_accessor< 0, extent<>, 3 > in;
        typedef inout_accessor< 1, extent<>, 3 > out;
        typedef boost::mpl::vector< in, out > arg_list;
        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation &eval, x_interval) {
            eval(out()) = call< copy_functor, x_interval >::at< -1, -1, 0 >::with(eval, in(1, 1, 0));
        }
    };

    struct call_call_copy_functor {
        typedef in_accessor< 0, extent<>, 3 > in;
        typedef inout_accessor< 1, extent<>, 3 > out;
        typedef boost::mpl::vector< in, out > arg_list;
        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation &eval, x_interval) {
            eval(out()) = call< call_copy_functor, x_interval >::with(eval, in());
        }
    };

    struct call_call_at_copy_functor {
        typedef in_accessor< 0, extent<>, 3 > in;
        typedef inout_accessor< 1, extent<>, 3 > out;
        typedef boost::mpl::vector< in, out > arg_list;
        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation &eval, x_interval) {
            eval(out()) = call< call_at_copy_functor, x_interval >::with(eval, in());
        }
    };

    struct call_call_with_offsets_copy_functor {
        typedef in_accessor< 0, extent<>, 3 > in;
        typedef inout_accessor< 1, extent<>, 3 > out;
        typedef boost::mpl::vector< in, out > arg_list;
        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation &eval, x_interval) {
            eval(out()) = call< call_with_offsets_copy_functor, x_interval >::with(eval, in());
        }
    };

    struct call_at_call_copy_functor {
        typedef in_accessor< 0, extent<>, 3 > in;
        typedef inout_accessor< 1, extent<>, 3 > out;
        typedef boost::mpl::vector< in, out > arg_list;
        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation &eval, x_interval) {
            eval(out()) = call< call_copy_functor, x_interval >::at< 1, 1, 0 >::with(eval, in());
        }
    };

    struct call_at_call_at_copy_functor {
        typedef in_accessor< 0, extent<>, 3 > in;
        typedef inout_accessor< 1, extent<>, 3 > out;
        typedef boost::mpl::vector< in, out > arg_list;
        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation &eval, x_interval) {
            eval(out()) = call< call_at_copy_functor, x_interval >::at< -1, -1, 0 >::with(eval, in());
        }
    };

    struct call_with_offsets_call_at_copy_functor {
        typedef in_accessor< 0, extent<>, 3 > in;
        typedef inout_accessor< 1, extent<>, 3 > out;
        typedef boost::mpl::vector< in, out > arg_list;
        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation &eval, x_interval) {
            eval(out()) = call< call_at_copy_functor, x_interval >::with(eval, in(-1, -1, 0));
        }
    };

    struct call_at_call_with_offsets_copy_functor {
        typedef in_accessor< 0, extent<>, 3 > in;
        typedef inout_accessor< 1, extent<>, 3 > out;
        typedef boost::mpl::vector< in, out > arg_list;
        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation &eval, x_interval) {
            eval(out()) = call< call_with_offsets_copy_functor, x_interval >::at< -1, -1, 0 >::with(eval, in());
        }
    };

    struct call_with_offsets_call_with_offsets_copy_functor {
        typedef in_accessor< 0, extent<>, 3 > in;
        typedef inout_accessor< 1, extent<>, 3 > out;
        typedef boost::mpl::vector< in, out > arg_list;
        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation &eval, x_interval) {
            eval(out()) = call< call_with_offsets_copy_functor, x_interval >::with(eval, in(-1, -1, 0));
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

    typedef gridtools::storage_traits< BACKEND::s_backend_id >::storage_info_t< 0, 3 > storage_info_t;
    typedef gridtools::storage_traits< BACKEND::s_backend_id >::data_store_t< float_type, storage_info_t > data_store_t;

    storage_info_t meta_;

    halo_descriptor di;
    halo_descriptor dj;
    gridtools::grid< call_interface_functors::axis > grid;

    verifier verifier_;
    array< array< uint_t, 2 >, 3 > verifier_halos;

    data_store_t in;
    data_store_t out;

    static constexpr float_type default_value = -1;
    data_store_t reference_unchanged;
    data_store_t reference_shifted;
    data_store_t reference_smaller_interval;

    typedef arg< 0, data_store_t > p_in;
    typedef arg< 1, data_store_t > p_out;
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
          verifier_halos{{{halo_size, halo_size}, {halo_size, halo_size}, {halo_size, halo_size}}},
          in(meta_, [](int i, int j, int k) { return i * 100 + j * 10 + k; }), out(meta_, default_value),
          reference_unchanged(meta_, [](int i, int j, int k) { return i * 100 + j * 10 + k; }),
          reference_shifted(meta_, [](int i, int j, int k) { return (i + 1) * 100 + (j + 1) * 10 + k; }),
          reference_smaller_interval(meta_,
              [this](int i, int j, int k) {
                  if (k > 0 && k < this->d3 - 1)
                      return (float_type)(i * 100 + j * 10 + k);
                  else
                      return default_value;
              }),
          domain(in, out) {
        grid.value_list[0] = 0;
        grid.value_list[1] = d3 - 1;
    }

    template < typename Computation >
    void execute_computation(Computation &comp) {
        comp->ready();
        comp->steady();
        comp->run();
        out.sync();
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

TEST_F(call_interface, call_to_copy_functor_with_out_first) {
    auto comp = gridtools::make_computation< gridtools::BACKEND >(
        domain,
        grid,
        gridtools::make_multistage(execute< forward >(),
            gridtools::make_stage< call_interface_functors::call_copy_functor_with_out_first >(p_in(), p_out())));

    execute_computation(comp);

    ASSERT_TRUE(verifier_.verify(grid, reference_unchanged, out, verifier_halos));
}

TEST_F(call_interface, call_to_copy_functor_with_expression) {
    auto comp = gridtools::make_computation< gridtools::BACKEND >(
        domain,
        grid,
        gridtools::make_multistage(execute< forward >(),
            gridtools::make_stage< call_interface_functors::call_copy_functor_with_expression >(p_in(), p_out())));

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

TEST_F(call_interface, call_to_copy_functor_default_interval) {
    auto comp = gridtools::make_computation< gridtools::BACKEND >(
        domain,
        grid,
        gridtools::make_multistage(execute< forward >(),
            gridtools::make_stage< call_interface_functors::call_copy_functor_default_interval >(p_in(), p_out())));

    execute_computation(comp);

    ASSERT_TRUE(verifier_.verify(grid, reference_unchanged, out, verifier_halos));
}

TEST_F(call_interface, call_to_copy_functor_default_interval_from_smaller_interval) {
    auto comp = gridtools::make_computation< gridtools::BACKEND >(
        domain,
        grid,
        gridtools::make_multistage(execute< forward >(),
            gridtools::make_stage< call_interface_functors::call_copy_functor_default_interval_from_smaller_interval >(
                                       p_in(), p_out())));

    execute_computation(comp);

    ASSERT_TRUE(verifier_.verify(grid, reference_smaller_interval, out, verifier_halos));
}

TEST_F(call_interface, call_to_copy_functor_default_interval_with_offset_in_k) {
    auto comp = gridtools::make_computation< gridtools::BACKEND >(
        domain,
        grid,
        gridtools::make_multistage(execute< forward >(),
            gridtools::make_stage< call_interface_functors::call_copy_functor_default_interval_with_offset_in_k >(
                                       p_in(), p_out())));

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

    typedef gridtools::storage_traits< BACKEND::s_backend_id >::storage_info_t< 0, 3 > storage_info_t;
    typedef gridtools::storage_traits< BACKEND::s_backend_id >::data_store_t< float_type, storage_info_t > data_store_t;

    storage_info_t meta_;

    halo_descriptor di;
    halo_descriptor dj;
    gridtools::grid< call_interface_functors::axis > grid;

    verifier verifier_;
    array< array< uint_t, 2 >, 3 > verifier_halos;

    data_store_t in;
    data_store_t out1;
    data_store_t out2;
    data_store_t reference_unchanged;
    data_store_t reference_shifted;

    typedef arg< 0, data_store_t > p_in;
    typedef arg< 1, data_store_t > p_out1;
    typedef arg< 2, data_store_t > p_out2;
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
          verifier_halos{{{halo_size, halo_size}, {halo_size, halo_size}, {halo_size, halo_size}}},
          in(meta_, [](int i, int j, int k) { return i * 100 + j * 10 + k; }), out1(meta_, -5), out2(meta_, -5),
          reference_unchanged(meta_, [](int i, int j, int k) { return i * 100 + j * 10 + k; }),
          reference_shifted(meta_, [](int i, int j, int k) { return (i + 1) * 100 + (j + 1) * 10 + k; }),
          domain(in, out1, out2) {
        grid.value_list[0] = 0;
        grid.value_list[1] = d3 - 1;
    }

    template < typename Computation >
    void execute_computation(Computation &comp) {
        comp->ready();
        comp->steady();
        comp->run();
        out1.sync();
        out2.sync();
    }
};

namespace call_proc_interface_functors {
    using axis = interval< level< 0, -1 >, level< 1, 1 > >;
    using x_interval = interval< level< 0, -1 >, level< 1, -1 > >;

    struct copy_functor {
        typedef in_accessor< 0, extent<>, 3 > in;
        typedef inout_accessor< 1, extent<>, 3 > out;
        typedef boost::mpl::vector< in, out > arg_list;
        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation &eval, x_interval) {
            eval(out()) = eval(in());
        }
    };

    struct copy_functor_with_expression {
        typedef in_accessor< 0, extent<>, 3 > in;
        typedef inout_accessor< 1, extent<>, 3 > out;
        typedef boost::mpl::vector< in, out > arg_list;
        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation &eval, x_interval) {
            eval(out()) = eval(in() + 0.);
        }
    };

    struct call_copy_functor_with_expression {
        typedef in_accessor< 0, extent<>, 3 > in;
        typedef inout_accessor< 1, extent<>, 3 > out;
        typedef boost::mpl::vector< in, out > arg_list;
        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation &eval, x_interval) {
            call_proc< copy_functor_with_expression, x_interval >::with(eval, in(), out());
        }
    };

    struct copy_twice_functor {
        typedef in_accessor< 0, extent<>, 3 > in;
        typedef inout_accessor< 1, extent<>, 3 > out1;
        typedef inout_accessor< 2, extent<>, 3 > out2;
        typedef boost::mpl::vector< in, out1, out2 > arg_list;
        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation &eval, x_interval) {
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
        GT_FUNCTION static void Do(Evaluation &eval, x_interval) {
            call_proc< copy_twice_functor, x_interval >::with(eval, in(), out1(), out2());
        }
    };

    struct call_with_offsets_copy_twice_functor {
        typedef in_accessor< 0, extent<>, 3 > in;
        typedef inout_accessor< 1, extent<>, 3 > out1;
        typedef inout_accessor< 2, extent<>, 3 > out2;
        typedef boost::mpl::vector< in, out1, out2 > arg_list;
        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation &eval, x_interval) {
            call_proc< copy_twice_functor, x_interval >::with(eval, in(1, 1, 0), out1(), out2());
        }
    };

    struct call_at_with_offsets_copy_twice_functor {
        typedef in_accessor< 0, extent<>, 3 > in;
        typedef inout_accessor< 1, extent<>, 3 > out1;
        typedef inout_accessor< 2, extent<>, 3 > out2;
        typedef boost::mpl::vector< in, out1, out2 > arg_list;
        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation &eval, x_interval) {
            call_proc< copy_twice_functor, x_interval >::at< 1, 1, 0 >::with(
                eval, in(), out1(-1, -1, 0), out2(-1, -1, 0)); // outs are at the original position
        }
    };

    struct copy_functor_default_interval {
        typedef in_accessor< 0, extent<>, 3 > in;
        typedef inout_accessor< 1, extent<>, 3 > out;
        typedef boost::mpl::vector< in, out > arg_list;
        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation &eval) {
            eval(out()) = eval(in());
        }
    };

    struct call_copy_functor_default_interval {
        typedef in_accessor< 0, extent<>, 3 > in;
        typedef inout_accessor< 1, extent<>, 3 > out;
        typedef boost::mpl::vector< in, out > arg_list;
        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation &eval) {
            call_proc< copy_functor_default_interval >::with(eval, in(), out());
        }
    };

    struct call_copy_functor_default_interval_with_offset_in_k {
        typedef in_accessor< 0, extent<>, 3 > in;
        typedef inout_accessor< 1, extent<>, 3 > out;
        typedef boost::mpl::vector< in, out > arg_list;
        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation &eval, x_interval) {
            call_proc< copy_functor_default_interval >::at< 0, 0, -1 >::with(eval, in(0, 0, 1), out(0, 0, 1));
        }
    };

    struct call_call_copy_twice_functor {
        typedef in_accessor< 0, extent<>, 3 > in;
        typedef inout_accessor< 1, extent<>, 3 > out1;
        typedef inout_accessor< 2, extent<>, 3 > out2;
        typedef boost::mpl::vector< in, out1, out2 > arg_list;
        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation &eval, x_interval) {
            call_proc< call_copy_twice_functor, x_interval >::with(eval, in(), out1(), out2());
        }
    };

    struct call_with_offsets_call_copy_twice_functor {
        typedef in_accessor< 0, extent<>, 3 > in;
        typedef inout_accessor< 1, extent<>, 3 > out1;
        typedef inout_accessor< 2, extent<>, 3 > out2;
        typedef boost::mpl::vector< in, out1, out2 > arg_list;
        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation &eval, x_interval) {
            call_proc< call_copy_twice_functor, x_interval >::with(eval, in(1, 1, 0), out1(), out2());
        }
    };

    struct call_with_offsets_call_with_offsets_copy_twice_functor {
        typedef in_accessor< 0, extent<>, 3 > in;
        typedef inout_accessor< 1, extent<>, 3 > out1;
        typedef inout_accessor< 2, extent<>, 3 > out2;
        typedef boost::mpl::vector< in, out1, out2 > arg_list;
        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation &eval, x_interval) {
            call_proc< call_with_offsets_copy_twice_functor, x_interval >::with(eval, in(-1, -1, 0), out1(), out2());
        }
    };

    struct call_with_local_variable {
        typedef in_accessor< 0, extent<>, 3 > in;
        typedef inout_accessor< 1, extent<>, 3 > out1;
        typedef inout_accessor< 2, extent<>, 3 > out2;
        typedef boost::mpl::vector< in, out1, out2 > arg_list;
        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation &eval, x_interval) {
            double local_in = 1;
            double local_out = -1;

            call_proc< copy_functor, x_interval >::with(eval, local_in, local_out);

            if (local_out > 0.) {
                eval(out1()) = eval(in());
            }
        }
    };
}

TEST_F(call_proc_interface, call_to_copy_functor_with_expression) {
    auto comp = gridtools::make_computation< gridtools::BACKEND >(
        domain,
        grid,
        gridtools::make_multistage(execute< forward >(),
            gridtools::make_stage< call_proc_interface_functors::call_copy_functor_with_expression >(
                                       p_in(), p_out1())));

    execute_computation(comp);

    ASSERT_TRUE(verifier_.verify(grid, reference_unchanged, out1, verifier_halos));
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

TEST_F(call_proc_interface, call_to_copy_functor_default_interval) {
    auto comp = gridtools::make_computation< gridtools::BACKEND >(
        domain,
        grid,
        gridtools::make_multistage(execute< forward >(),
            gridtools::make_stage< call_proc_interface_functors::call_copy_functor_default_interval >(
                                       p_in(), p_out1())));

    execute_computation(comp);

    ASSERT_TRUE(verifier_.verify(grid, reference_unchanged, out1, verifier_halos));
}

TEST_F(call_proc_interface, call_to_copy_functor_default_interval_with_offset_in_k) {
    auto comp = gridtools::make_computation< gridtools::BACKEND >(
        domain,
        grid,
        gridtools::make_multistage(execute< forward >(),
            gridtools::make_stage< call_proc_interface_functors::call_copy_functor_default_interval_with_offset_in_k >(
                                       p_in(), p_out1())));

    execute_computation(comp);

    ASSERT_TRUE(verifier_.verify(grid, reference_unchanged, out1, verifier_halos));
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
