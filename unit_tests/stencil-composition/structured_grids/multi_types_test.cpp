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
#include <gridtools.hpp>
#include <stencil-composition/stencil-composition.hpp>
#include <stencil-composition/structured_grids/call_interfaces.hpp>
#include <tools/verifier.hpp>
#include "gtest/gtest.h"

#ifdef __CUDACC__
#ifdef FUNCTIONS_CALL
#define FTESTNAME(x) CALL_GPU
#endif

#ifdef FUNCTIONS_OFFSETS
#define FTESTNAME(x) OFFSETS_GPU
#endif

#ifdef FUNCTIONS_PROCEDURES
#define FTESTNAME(x) PROCEDURES_GPU
#endif

#ifdef FUNCTIONS_PROCEDURES_OFFSETS
#define FTESTNAME(x) PROCEDURESOFFSETS_GPU
#endif
#else
#ifdef FUNCTIONS_CALL
#define FTESTNAME(x) CALL
#endif

#ifdef FUNCTIONS_OFFSETS
#define FTESTNAME(x) OFFSETS
#endif

#ifdef FUNCTIONS_PROCEDURES
#define FTESTNAME(x) PROCEDURES
#endif

#ifdef FUNCTIONS_PROCEDURES_OFFSETS
#define FTESTNAME(x) PROCEDURESOFFSETS
#endif
#endif

namespace multi_types_test {
    using gridtools::level;
    using gridtools::accessor;
    using gridtools::extent;
    using gridtools::arg;

    using namespace gridtools;
    using namespace enumtype;
    using namespace expressions;

#ifdef CUDA_EXAMPLE
#define BACKEND backend< Cuda, GRIDBACKEND, Block >
#else
#ifdef BACKEND_BLOCK
#define BACKEND backend< Host, GRIDBACKEND, Block >
#else
#define BACKEND backend< Host, GRIDBACKEND, Naive >
#endif
#endif

    typedef gridtools::interval< level< 0, -1 >, level< 1, -1 > > region;

    typedef gridtools::interval< level< 0, -2 >, level< 1, 3 > > axis;

    struct type4;

    struct type1 {
        int i, j, k;

        GT_FUNCTION
        type1() : i(0), j(0), k(0) {}
        GT_FUNCTION
        explicit type1(int i, int j, int k) : i(i), j(j), k(k) {}
    };

    struct type4 {
        float x, y, z;

        GT_FUNCTION
        type4() : x(0.), y(0.), z(0.) {}
        GT_FUNCTION
        explicit type4(double i, double j, double k) : x(i), y(j), z(k) {}

        GT_FUNCTION
        type4 &operator=(type1 const &a) {
            x = a.i;
            y = a.j;
            z = a.k;
            return *this;
        }
    };

    struct type2 {
        double xy;
        GT_FUNCTION
        type2 &operator=(type4 const &x) {
            xy = x.x + x.y;
            return *this;
        }
    };

    struct type3 {
        double yz;

        GT_FUNCTION
        type3 &operator=(type4 const &x) {
            yz = x.y + x.z;
            return *this;
        }
    };

    GT_FUNCTION
    type4 operator+(type4 const &a, type1 const &b) {
        return type4(
            a.x + static_cast< double >(b.i), a.y + static_cast< double >(b.j), a.z + static_cast< double >(b.k));
    }

    GT_FUNCTION
    type4 operator-(type4 const &a, type1 const &b) {
        return type4(
            a.x - static_cast< double >(b.i), a.y - static_cast< double >(b.j), a.z - static_cast< double >(b.k));
    }

    GT_FUNCTION
    type4 operator+(type1 const &a, type4 const &b) {
        return type4(
            a.i + static_cast< double >(b.x), a.j + static_cast< double >(b.y), a.k + static_cast< double >(b.z));
    }

    GT_FUNCTION
    type4 operator-(type1 const &a, type4 const &b) {
        return type4(
            a.i - static_cast< double >(b.x), a.j - static_cast< double >(b.y), a.k - static_cast< double >(b.z));
    }

    GT_FUNCTION
    type4 operator+(type1 const &a, type1 const &b) {
        return type4(
            a.i + static_cast< double >(b.i), a.j + static_cast< double >(b.j), a.k + static_cast< double >(b.k));
    }

    GT_FUNCTION
    type4 operator-(type1 const &a, type1 const &b) {
        return type4(
            a.i - static_cast< double >(b.i), a.j - static_cast< double >(b.j), a.k - static_cast< double >(b.k));
    }

    struct function0 {
        typedef accessor< 0, enumtype::in > in;
        typedef accessor< 1, enumtype::inout > out;

        typedef boost::mpl::vector< in, out > arg_list;

        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation const &eval, region) {
            eval(out()).i = eval(in()).i + 1;
            eval(out()).j = eval(in()).j + 1;
            eval(out()).k = eval(in()).k + 1;
        }
    };

    struct function1 {
        typedef accessor< 0, enumtype::inout > out;
        typedef accessor< 1, enumtype::in > in;

        typedef boost::mpl::vector< out, in > arg_list;

        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation const &eval, region) {
#ifdef FUNCTIONS_PROCEDURES
            type1 result;
            call_proc< function0, region >::with(eval, in(), result);
#else
#ifdef FUNCTIONS_PROCEDURES_OFFSETS
            type1 result;
            call_proc< function0, region >::with_offsets(eval, in(), result);
#else
#ifdef FUNCTIONS_OFFSETS
            auto result = call< function0, region >::with_offsets(eval, in());
#else
            auto result = call< function0, region >::with(eval, in());
#endif
#endif
#endif
            eval(out()) = result;
        }
    };

    struct function2 {

        typedef accessor< 0, enumtype::inout > out;
        typedef accessor< 1, enumtype::in > in;
        typedef accessor< 2, enumtype::in > temp;

        typedef boost::mpl::vector< out, in, temp > arg_list;

        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation const &eval, region) {
            eval(out()) = eval(temp()) + eval(in());
            // std::cout << (eval(temp())+eval(in())).x << ", "
            //           << (eval(temp())+eval(in())).y << ", "
            //           << (eval(temp())+eval(in())).z << ": "
            //           << " " << (eval(out())).xy << std::endl;
        }
    };

    struct function3 {

        typedef accessor< 0, enumtype::inout > out;
        typedef accessor< 1, enumtype::in > temp;
        typedef accessor< 2, enumtype::in > in;

        typedef boost::mpl::vector< out, temp, in > arg_list;

        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation const &eval, region) {
            eval(out()) = eval(temp()) - eval(in());
        }
    };

    /*
     * The following operators and structs are for debugging only
     */
    std::ostream &operator<<(std::ostream &s, function1 const) { return s << "function1"; }
    std::ostream &operator<<(std::ostream &s, function2 const) { return s << "function2"; }
    std::ostream &operator<<(std::ostream &s, function3 const) { return s << "function3"; }

    bool test(uint_t x, uint_t y, uint_t z) {

        uint_t d1 = x;
        uint_t d2 = y;
        uint_t d3 = z;
        uint_t halo_size = 0;

        typedef gridtools::storage_traits< BACKEND::s_backend_id >::storage_info_t< 0, 3 > storage_info1_t;
        typedef gridtools::storage_traits< BACKEND::s_backend_id >::storage_info_t< 1, 3 > storage_info2_t;
        typedef gridtools::storage_traits< BACKEND::s_backend_id >::storage_info_t< 2, 3 > storage_info3_t;
        typedef gridtools::storage_traits< BACKEND::s_backend_id >::data_store_t< type1, storage_info1_t >
            data_store1_t;
        typedef gridtools::storage_traits< BACKEND::s_backend_id >::data_store_t< type2, storage_info2_t >
            data_store2_t;
        typedef gridtools::storage_traits< BACKEND::s_backend_id >::data_store_t< type3, storage_info3_t >
            data_store3_t;

        // TODO: Use storage_info as unnamed object - lifetime issues on GPUs
        storage_info1_t si1(x, y, z);
        storage_info2_t si2(x, y, z);
        storage_info3_t si3(x, y, z);

        data_store1_t field1 = data_store1_t(si1, [](int i, int j, int k) { return type1(i, j, k); });
        data_store2_t field2 = data_store2_t(si2, type2());
        data_store3_t field3 = data_store3_t(si3, type3());

        typedef tmp_arg< 3, data_store1_t > p_temp;
        typedef arg< 0, data_store1_t > p_field1;
        typedef arg< 1, data_store2_t > p_field2;
        typedef arg< 2, data_store3_t > p_field3;

        typedef boost::mpl::vector< p_field1, p_field2, p_field3, p_temp > accessor_list;

        gridtools::aggregator_type< accessor_list > domain(field1, field2, field3);

        uint_t di[5] = {halo_size, halo_size, halo_size, d1 - halo_size - 1, d1};
        uint_t dj[5] = {halo_size, halo_size, halo_size, d2 - halo_size - 1, d2};

        gridtools::grid< axis > grid(di, dj);
        grid.value_list[0] = 0;
        grid.value_list[1] = d3 - 1;

        auto test_computation = gridtools::make_computation< BACKEND >(
            domain,
            grid,
            gridtools::make_multistage // mss_descriptor
            (execute< forward >(),
                gridtools::make_stage< function1 >(p_temp(), p_field1()),
                gridtools::make_stage< function2 >(p_field2(), p_field1(), p_temp())),
            gridtools::make_multistage // mss_descriptor
            (execute< backward >(),
                gridtools::make_stage< function1 >(p_temp(), p_field1()),
                gridtools::make_stage< function3 >(p_field3(), p_temp(), p_field1())));

        test_computation->ready();

        test_computation->steady();

        test_computation->run();

        test_computation->finalize();

        auto f1v = make_host_view(field1);
        auto f2v = make_host_view(field2);
        auto f3v = make_host_view(field3);
        assert(check_consistency(field1, f1v) && "view cannot be used safely.");
        assert(check_consistency(field2, f2v) && "view cannot be used safely.");
        assert(check_consistency(field3, f3v) && "view cannot be used safely.");

        bool result = true;
        for (int i = 0; i < x; ++i) {
            for (int j = 0; j < y; ++j) {
                for (int k = 0; k < z; ++k) {
                    double xy =
                        static_cast< double >(2 * f1v(i, j, k).i + 1) + static_cast< double >(2 * f1v(i, j, k).j + 1);
                    double yz = 2;
                    if (f2v(i, j, k).xy != xy) {
                        result = false;
                        std::cout << "(" << i << ", " << j << ", " << k << ") : " << f2v(i, j, k).xy << " != " << xy
                                  << " diff = " << f2v(i, j, k).xy - xy << std::endl;
                    }
                    if (f3v(i, j, k).yz != yz) {
                        result = false;
                        std::cout << "(" << i << ", " << j << ", " << k << ") : " << f3v(i, j, k).yz << " != " << yz
                                  << " diff = " << f3v(i, j, k).yz - yz << std::endl;
                    }
                }
            }
        }

        if (!result) {
            std::cout << "ERROR" << std::endl;
        }

        return result;
    }
} // namespace multi_types_test

TEST(multitypes, FTESTNAME(x)) { EXPECT_TRUE(multi_types_test::test(4, 4, 4)); }
