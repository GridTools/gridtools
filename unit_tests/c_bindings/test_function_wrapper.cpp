/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gridtools/c_bindings/function_wrapper.hpp>

#include <iostream>
#include <stack>
#include <type_traits>

#include <gtest/gtest.h>

#include <gridtools/c_bindings/handle.h>

namespace gridtools {
    namespace c_bindings {
        namespace {

            struct a_struct {};
            struct array_descriptor_struct {
                array_descriptor_struct(const gt_fortran_array_descriptor &);
                using gt_view_element_type = int;
                using gt_view_rank = std::integral_constant<std::size_t, 3>;
            };
            static_assert(std::is_same<wrapped_t<void (*)()>, void()>::value, "");
            static_assert(std::is_same<wrapped_t<int()>, int()>::value, "");
            static_assert(std::is_same<wrapped_t<a_struct()>, gt_handle *()>::value, "");
            static_assert(std::is_same<wrapped_t<a_struct const &()>, gt_handle *()>::value, "");
            static_assert(std::is_same<wrapped_t<void(int)>, void(int)>::value, "");
            static_assert(std::is_same<wrapped_t<void(int &)>, void(int *)>::value, "");
            static_assert(std::is_same<wrapped_t<void(int const *)>, void(int const *)>::value, "");
            static_assert(std::is_same<wrapped_t<void(a_struct *)>, void(gt_handle *)>::value, "");
            static_assert(std::is_same<wrapped_t<void(a_struct &)>, void(gt_handle *)>::value, "");
            static_assert(std::is_same<wrapped_t<void(a_struct)>, void(gt_handle *)>::value, "");
            static_assert(
                std::is_same<wrapped_t<void(float (&)[1][2][3])>, void(gt_fortran_array_descriptor *)>::value, "");
            static_assert(std::is_same<wrapped_t<array_descriptor_struct(array_descriptor_struct)>,
                              gt_handle *(gt_fortran_array_descriptor *)>::value,
                "");

            template <class T>
            std::stack<T> create() {
                return std::stack<T>{};
            }

            template <class T>
            void push_to_ref(std::stack<T> &obj, T val) {
                obj.push(val);
            }

            template <class T>
            void push_to_ptr(std::stack<T> *obj, T val) {
                obj->push(val);
            }

            template <class T>
            void pop(std::stack<T> &obj) {
                obj.pop();
            }

            template <class T>
            T top(const std::stack<T> &obj) {
                return obj.top();
            }

            template <class T>
            bool empty(const std::stack<T> &obj) {
                return obj.empty();
            }

            TEST(wrap, smoke) {
                gt_handle *obj = wrap(create<int>)();
                EXPECT_TRUE(wrap(empty<int>)(obj));
                wrap(push_to_ref<int>)(obj, 42);
                EXPECT_FALSE(wrap(empty<int>)(obj));
                EXPECT_EQ(42, wrap(top<int>)(obj));
                wrap(push_to_ptr<int>)(obj, 43);
                EXPECT_EQ(43, wrap(top<int>)(obj));
                wrap(pop<int>)(obj);
                wrap(pop<int>)(obj);
                EXPECT_TRUE(wrap(empty<int>)(obj));
                gt_release(obj);
            }

            std::unique_ptr<int> make_ptr() { return std::unique_ptr<int>{new int{3}}; }
            std::unique_ptr<int> forward_ptr(std::unique_ptr<int> &&ptr) { return std::move(ptr); }
            void set_ptr(std::unique_ptr<int> &ptr, int v) { *ptr = v; }
            int get_ptr(std::unique_ptr<int> &ptr) { return *ptr; }
            bool is_ptr_set(std::unique_ptr<int> &ptr) { return ptr.get(); }
            TEST(wrap, return_values) {
                gt_handle *obj = wrap(make_ptr)();
                wrap(set_ptr)(obj, 3);
                EXPECT_EQ(3, wrap(get_ptr)(obj));
                gt_handle *obj2 = wrap(forward_ptr)(obj);
                wrap(set_ptr)(obj2, 4);
                EXPECT_EQ(4, wrap(get_ptr)(obj2));
                EXPECT_FALSE(wrap(is_ptr_set)(obj));
                gt_release(obj);
                gt_release(obj2);
            }

            void inc(int &val) { ++val; }

            TEST(wrap, const_expr) {
                constexpr auto wrapped_inc = wrap(inc);
                int i = 41;
                wrapped_inc(&i);
                EXPECT_EQ(42, i);
            }

            TEST(wrap, lambda) {
                EXPECT_EQ(42, wrap(+[] { return 42; })());
            }

            TEST(wrap, lambda2) {
                int val = 42;
                auto testee = wrap<int()>([&] { return val; });
                EXPECT_EQ(42, testee());
                val = 1;
                EXPECT_EQ(1, testee());
            }

            TEST(wrap, array_descriptor) {
                int array[2][3] = {{1, 2, 3}, {4, 5, 6}};
                gt_fortran_array_descriptor descriptor;
                descriptor.data = array;
                descriptor.type = gt_fk_Int;
                descriptor.rank = 2;
                descriptor.dims[0] = 3;
                descriptor.dims[1] = 2;
                descriptor.is_acc_present = false;

                auto get = wrap<int(int(&)[2][3], size_t, size_t)>(
                    [](int(&array)[2][3], size_t i, size_t j) { return array[i][j]; });
                EXPECT_EQ(array[0][0], get(&descriptor, 0, 0));
            }
        } // namespace
    }     // namespace c_bindings
} // namespace gridtools
