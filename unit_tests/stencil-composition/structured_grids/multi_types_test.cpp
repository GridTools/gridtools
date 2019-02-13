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
#include <gtest/gtest.h>

#include <gridtools/stencil-composition/stencil-composition.hpp>
#include <gridtools/stencil-composition/stencil-functions/stencil-functions.hpp>
#include <gridtools/tools/backend_select.hpp>
#include <gridtools/tools/verifier.hpp>

#ifdef __CUDACC__
#ifdef FUNCTIONS_CALL
#define FTESTNAME(x) CALL_GPU
#endif

#ifdef FUNCTIONS_PROCEDURES
#define FTESTNAME(x) PROCEDURES_GPU
#endif
#else
#ifdef FUNCTIONS_CALL
#define FTESTNAME(x) CALL
#endif

#ifdef FUNCTIONS_PROCEDURES
#define FTESTNAME(x) PROCEDURES
#endif
#endif

namespace multi_types_test {
    using gridtools::accessor;
    using gridtools::arg;
    using gridtools::extent;
    using gridtools::level;

    using namespace gridtools;
    using namespace execute;
    using namespace expressions;

    using axis_t = axis<1>;
    using region = axis<1>::full_interval;

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
        return type4(a.x + static_cast<double>(b.i), a.y + static_cast<double>(b.j), a.z + static_cast<double>(b.k));
    }

    GT_FUNCTION
    type4 operator-(type4 const &a, type1 const &b) {
        return type4(a.x - static_cast<double>(b.i), a.y - static_cast<double>(b.j), a.z - static_cast<double>(b.k));
    }

    GT_FUNCTION
    type4 operator+(type1 const &a, type4 const &b) {
        return type4(a.i + static_cast<double>(b.x), a.j + static_cast<double>(b.y), a.k + static_cast<double>(b.z));
    }

    GT_FUNCTION
    type4 operator-(type1 const &a, type4 const &b) {
        return type4(a.i - static_cast<double>(b.x), a.j - static_cast<double>(b.y), a.k - static_cast<double>(b.z));
    }

    GT_FUNCTION
    type4 operator+(type1 const &a, type1 const &b) {
        return type4(a.i + static_cast<double>(b.i), a.j + static_cast<double>(b.j), a.k + static_cast<double>(b.k));
    }

    GT_FUNCTION
    type4 operator-(type1 const &a, type1 const &b) {
        return type4(a.i - static_cast<double>(b.i), a.j - static_cast<double>(b.j), a.k - static_cast<double>(b.k));
    }

    struct function0 {
        typedef accessor<0, intent::in> in;
        typedef accessor<1, intent::inout> out;

        typedef make_param_list<in, out> param_list;

        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &eval, region) {
            eval(out()).i = eval(in()).i + 1;
            eval(out()).j = eval(in()).j + 1;
            eval(out()).k = eval(in()).k + 1;
        }
    };

    struct function1 {
        typedef accessor<0, intent::inout> out;
        typedef accessor<1, intent::in> in;

        typedef make_param_list<out, in> param_list;

        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &eval, region) {
#ifdef FUNCTIONS_PROCEDURES
            type1 result;
            call_proc<function0, region>::with(eval, in(), result);
            call_proc<function0, region>::with(eval, in(), result);
#else
            auto result = call<function0, region>::with(eval, in());
#endif
            eval(out()) = result;
        }
    };

    struct function2 {

        typedef accessor<0, intent::inout> out;
        typedef accessor<1, intent::in> in;
        typedef accessor<2, intent::in> temp;

        typedef make_param_list<out, in, temp> param_list;

        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &eval, region) {
            eval(out()) = eval(temp()) + eval(in());
        }
    };

    struct function3 {

        typedef accessor<0, intent::inout> out;
        typedef accessor<1, intent::in> temp;
        typedef accessor<2, intent::in> in;

        typedef make_param_list<out, temp, in> param_list;

        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &eval, region) {
            eval(out()) = eval(temp()) - eval(in());
        }
    };

    bool test(uint_t x, uint_t y, uint_t z) {

        uint_t d1 = x;
        uint_t d2 = y;
        uint_t d3 = z;
        uint_t halo_size = 0;

        typedef gridtools::storage_traits<backend_t::backend_id_t>::storage_info_t<0, 3> storage_info1_t;
        typedef gridtools::storage_traits<backend_t::backend_id_t>::storage_info_t<1, 3> storage_info2_t;
        typedef gridtools::storage_traits<backend_t::backend_id_t>::storage_info_t<2, 3> storage_info3_t;
        typedef gridtools::storage_traits<backend_t::backend_id_t>::data_store_t<type1, storage_info1_t> data_store1_t;
        typedef gridtools::storage_traits<backend_t::backend_id_t>::data_store_t<type2, storage_info2_t> data_store2_t;
        typedef gridtools::storage_traits<backend_t::backend_id_t>::data_store_t<type3, storage_info3_t> data_store3_t;

        // TODO: Use storage_info as unnamed object - lifetime issues on GPUs
        storage_info1_t si1(x, y, z);
        storage_info2_t si2(x, y, z);
        storage_info3_t si3(x, y, z);

        data_store1_t field1 = data_store1_t(si1, [](int i, int j, int k) { return type1(i, j, k); });
        data_store2_t field2 = data_store2_t(si2, type2());
        data_store3_t field3 = data_store3_t(si3, type3());

        typedef tmp_arg<3, data_store1_t> p_temp;
        typedef arg<0, data_store1_t> p_field1;
        typedef arg<1, data_store2_t> p_field2;
        typedef arg<2, data_store3_t> p_field3;

        halo_descriptor di{halo_size, halo_size, halo_size, d1 - halo_size - 1, d1};
        halo_descriptor dj{halo_size, halo_size, halo_size, d2 - halo_size - 1, d2};

        auto grid = make_grid(di, dj, axis_t(d3));

        auto test_computation = gridtools::make_computation<backend_t>(grid,
            p_field1() = field1,
            p_field2() = field2,
            p_field3() = field3,
            gridtools::make_multistage // mss_descriptor
            (execute::forward(),
                gridtools::make_stage<function1>(p_temp(), p_field1()),
                gridtools::make_stage<function2>(p_field2(), p_field1(), p_temp())),
            gridtools::make_multistage // mss_descriptor
            (execute::backward(),
                gridtools::make_stage<function1>(p_temp(), p_field1()),
                gridtools::make_stage<function3>(p_field3(), p_temp(), p_field1())));

        test_computation.run();

        test_computation.sync_bound_data_stores();

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
                        static_cast<double>(2 * f1v(i, j, k).i + 1) + static_cast<double>(2 * f1v(i, j, k).j + 1);
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

TEST(multitypes, FTESTNAME(x)) { EXPECT_TRUE(multi_types_test::test(4, 5, 6)); }
