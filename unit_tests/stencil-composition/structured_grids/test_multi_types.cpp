/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <iostream>

#include <gtest/gtest.h>

#include <gridtools/stencil-composition/stencil-composition.hpp>
#include <gridtools/stencil-composition/stencil-functions/stencil-functions.hpp>
#include <gridtools/tools/computation_fixture.hpp>

namespace gridtools {
    namespace {
        struct type1 {
            int i, j, k;

            GT_FUNCTION type1() : i(0), j(0), k(0) {}
            GT_FUNCTION type1(int i, int j, int k) : i(i), j(j), k(k) {}
        };

        struct type4 {
            float x, y, z;

            GT_FUNCTION type4() : x(0.), y(0.), z(0.) {}
            GT_FUNCTION type4(double i, double j, double k) : x(i), y(j), z(k) {}

            GT_FUNCTION type4 &operator=(type1 const &a) {
                x = a.i;
                y = a.j;
                z = a.k;
                return *this;
            }
        };

        struct type2 {
            double xy;
            GT_FUNCTION type2 &operator=(type4 x) {
                xy = x.x + x.y;
                return *this;
            }
            friend std::ostream &operator<<(std::ostream &strm, type2 obj) {
                return strm << "{ xy: " << obj.xy << " }";
            }
            friend bool operator==(type2 lhs, type2 rhs) { return lhs.xy == rhs.xy; }
        };

        struct type3 {
            double yz;

            GT_FUNCTION type3 &operator=(type4 x) {
                yz = x.y + x.z;
                return *this;
            }
            friend std::ostream &operator<<(std::ostream &strm, type3 obj) {
                return strm << "{ yz: " << obj.yz << " }";
            }
            friend bool operator==(type3 lhs, type3 rhs) { return lhs.yz == rhs.yz; }
        };

        GT_FUNCTION type4 operator+(type4 a, type1 b) {
            return {a.x + double(b.i), a.y + double(b.j), a.z + double(b.k)};
        }

        GT_FUNCTION type4 operator-(type4 a, type1 b) {
            return {a.x - double(b.i), a.y - double(b.j), a.z - double(b.k)};
        }

        GT_FUNCTION type4 operator+(type1 a, type4 b) {
            return {a.i + double(b.x), a.j + double(b.y), a.k + double(b.z)};
        }

        GT_FUNCTION type4 operator-(type1 a, type4 b) {
            return {a.i - double(b.x), a.j - double(b.y), a.k - double(b.z)};
        }

        GT_FUNCTION type4 operator+(type1 a, type1 b) {
            return {a.i + double(b.i), a.j + double(b.j), a.k + double(b.k)};
        }

        GT_FUNCTION type4 operator-(type1 a, type1 b) {
            return {a.i - double(b.i), a.j - double(b.j), a.k - double(b.k)};
        }

        struct function0 {
            using in = in_accessor<0>;
            using out = inout_accessor<1>;

            using param_list = make_param_list<in, out>;

            template <typename Evaluation>
            GT_FUNCTION static void apply(Evaluation &eval) {
                eval(out()).i = eval(in()).i + 1;
                eval(out()).j = eval(in()).j + 1;
                eval(out()).k = eval(in()).k + 1;
            }
        };

        enum class call_type { function, procedure };

        template <call_type Type, class Eval, class T, enable_if_t<Type == call_type::function, int> = 0>
        GT_FUNCTION auto call_function0(Eval &eval, T obj) GT_AUTO_RETURN(call<function0>::with(eval, obj));

        template <call_type Type, class Eval, class T, enable_if_t<Type == call_type::procedure, int> = 0>
        GT_FUNCTION type1 call_function0(Eval &eval, T obj) {
            type1 res;
            call_proc<function0>::with(eval, obj, res);
            return res;
        }

        template <call_type CallType>
        struct function1 {
            using out = inout_accessor<0>;
            using in = in_accessor<1>;

            using param_list = make_param_list<out, in>;

            template <typename Evaluation>
            GT_FUNCTION static void apply(Evaluation &eval) {
                eval(out()) = call_function0<CallType>(eval, in());
            }
        };

        struct function2 {
            using out = inout_accessor<0>;
            using in = in_accessor<1>;
            using temp = in_accessor<2>;

            using param_list = make_param_list<out, in, temp>;

            template <typename Evaluation>
            GT_FUNCTION static void apply(Evaluation &eval) {
                eval(out()) = eval(temp()) + eval(in());
            }
        };

        struct function3 {
            using out = inout_accessor<0>;
            using temp = in_accessor<1>;
            using in = in_accessor<2>;

            using param_list = make_param_list<out, temp, in>;

            template <typename Evaluation>
            GT_FUNCTION static void apply(Evaluation &eval) {
                eval(out()) = eval(temp()) - eval(in());
            }
        };

        struct multitypes : computation_fixture<> {
            multitypes() : computation_fixture<>{4, 5, 6} {}

            using storage_type1 = storage_tr::data_store_t<type1, storage_info_t>;
            using storage_type2 = storage_tr::data_store_t<type2, storage_info_t>;
            using storage_type3 = storage_tr::data_store_t<type3, storage_info_t>;

            arg<1, storage_type1> p_field1;
            arg<2, storage_type2> p_field2;
            arg<3, storage_type3> p_field3;
            tmp_arg<0, storage_type1> p_temp;

            template <call_type CallType>
            void do_test() {
                auto in = [](int i, int j, int k) -> type1 { return {i, j, k}; };
                auto field1 = make_storage<storage_type1>(in);
                auto field2 = make_storage<storage_type2>();
                auto field3 = make_storage<storage_type3>();

                using fun1 = function1<CallType>;

                make_computation(p_field1 = make_storage<storage_type1>(in),
                    p_field2 = field2,
                    p_field3 = field3,
                    make_multistage(execute::forward(),
                        make_stage<fun1>(p_temp, p_field1),
                        make_stage<function2>(p_field2, p_field1, p_temp)),
                    make_multistage(execute::backward(),
                        make_stage<fun1>(p_temp, p_field1),
                        make_stage<function3>(p_field3, p_temp, p_field1)))
                    .run();

                verify(make_storage<storage_type2>([&in](int i, int j, int k) -> type2 {
                    auto f1 = in(i, j, k);
                    return {2. * (f1.i + f1.j + 1)};
                }),
                    field2);

                verify(make_storage<storage_type3>([](int i, int j, int k) -> type3 { return {2}; }), field3);
            }
        };

        TEST_F(multitypes, function) { do_test<call_type::function>(); }

        TEST_F(multitypes, procedure) { do_test<call_type::procedure>(); }

    } // namespace
} // namespace gridtools
