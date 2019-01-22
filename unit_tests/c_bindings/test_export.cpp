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

#include <gridtools/c_bindings/export.hpp>

#include <functional>
#include <sstream>
#include <stack>

#include <gtest/gtest.h>

#include <gridtools/c_bindings/handle.h>

namespace {

    using stack_t = std::stack<double>;

    // Various flavours to create exported functions.

    stack_t my_create_impl() { return stack_t{}; }
    GT_EXPORT_BINDING_0(my_create, my_create_impl);

    template <class T>
    void push_impl(std::stack<T> *obj, T val) {
        obj->push(val);
    }
    GT_EXPORT_GENERIC_BINDING(2, my_push, push_impl, (float)(int)(double));

    GT_EXPORT_BINDING_WITH_SIGNATURE_1(my_pop, void(stack_t &), [](stack_t &obj) { obj.pop(); });

    struct top_impl {
        template <class T>
        T operator()(std::stack<T> const &container) const {
            return container.top();
        }
    };
    GT_EXPORT_BINDING_WITH_SIGNATURE_1(my_top, double(stack_t const &), top_impl{});

    GT_EXPORT_BINDING_WITH_SIGNATURE_1(my_empty, bool(stack_t const &), std::mem_fn(&stack_t::empty));

    template <class T, size_t size>
    void assign_impl(T (&obj)[size][size], T val) {
        for (size_t i = 0; i < size; ++i) {
            for (size_t j = 0; j < size; ++j) {
                obj[i][j] = val;
            }
        }
    }
    GT_EXPORT_GENERIC_BINDING_WRAPPED(2, my_assign, assign_impl, (int, 2)(double, 2));

    struct c_bindings_compatible_type {
        c_bindings_compatible_type(const gt_fortran_array_descriptor &) {}
    };
    struct wrapper_compatible_type {
        wrapper_compatible_type(const gt_fortran_array_descriptor &) {}
    };
    gt_fortran_array_descriptor get_fortran_view_meta(wrapper_compatible_type *) {
        gt_fortran_array_descriptor d;
        d.rank = 2;
        d.type = gt_fk_Int;
        d.is_acc_present = false;
        return d;
    }
    void test_c_bindings_and_wrapper_compatible_type_impl(c_bindings_compatible_type, wrapper_compatible_type) {}
    GT_EXPORT_BINDING_2(
        test_c_bindings_and_wrapper_compatible_type_a, test_c_bindings_and_wrapper_compatible_type_impl);
    GT_EXPORT_BINDING_WRAPPED_2(
        test_c_bindings_and_wrapper_compatible_type_b, test_c_bindings_and_wrapper_compatible_type_impl);

    TEST(export, smoke) {
        gt_handle *obj = my_create();
        EXPECT_TRUE(my_empty(obj));
        my_push2(obj, 42);
        EXPECT_FALSE(my_empty(obj));
        EXPECT_EQ(42, my_top(obj));
        my_pop(obj);
        EXPECT_TRUE(my_empty(obj));
        gt_release(obj);
    }

    const char expected_c_interface[] = R"?(// This file is generated!
#pragma once

#include <gridtools/c_bindings/array_descriptor.h>
#include <gridtools/c_bindings/handle.h>

#ifdef __cplusplus
extern "C" {
#endif

void my_assign0(gt_fortran_array_descriptor*, int);
void my_assign1(gt_fortran_array_descriptor*, double);
gt_handle* my_create();
bool my_empty(gt_handle*);
void my_pop(gt_handle*);
void my_push0(gt_handle*, float);
void my_push1(gt_handle*, int);
void my_push2(gt_handle*, double);
double my_top(gt_handle*);
void test_c_bindings_and_wrapper_compatible_type_a(gt_fortran_array_descriptor*, gt_fortran_array_descriptor*);
void test_c_bindings_and_wrapper_compatible_type_b(gt_fortran_array_descriptor*, gt_fortran_array_descriptor*);

#ifdef __cplusplus
}
#endif
)?";

    TEST(export, c_interface) {
        std::ostringstream strm;
        gridtools::c_bindings::generate_c_interface(strm);
        EXPECT_EQ(strm.str(), expected_c_interface);
    }

    const char expected_fortran_interface[] = R"?(! This file is generated!
module my_module
implicit none
  interface

    subroutine my_assign0_impl(arg0, arg1) bind(c, name="my_assign0")
      use iso_c_binding
      use array_descriptor
      type(gt_fortran_array_descriptor) :: arg0
      integer(c_int), value :: arg1
    end subroutine
    subroutine my_assign1_impl(arg0, arg1) bind(c, name="my_assign1")
      use iso_c_binding
      use array_descriptor
      type(gt_fortran_array_descriptor) :: arg0
      real(c_double), value :: arg1
    end subroutine
    type(c_ptr) function my_create() bind(c)
      use iso_c_binding
    end function
    logical(c_bool) function my_empty(arg0) bind(c)
      use iso_c_binding
      type(c_ptr), value :: arg0
    end function
    subroutine my_pop(arg0) bind(c)
      use iso_c_binding
      type(c_ptr), value :: arg0
    end subroutine
    subroutine my_push0(arg0, arg1) bind(c)
      use iso_c_binding
      type(c_ptr), value :: arg0
      real(c_float), value :: arg1
    end subroutine
    subroutine my_push1(arg0, arg1) bind(c)
      use iso_c_binding
      type(c_ptr), value :: arg0
      integer(c_int), value :: arg1
    end subroutine
    subroutine my_push2(arg0, arg1) bind(c)
      use iso_c_binding
      type(c_ptr), value :: arg0
      real(c_double), value :: arg1
    end subroutine
    real(c_double) function my_top(arg0) bind(c)
      use iso_c_binding
      type(c_ptr), value :: arg0
    end function
    subroutine test_c_bindings_and_wrapper_compatible_type_a(arg0, arg1) bind(c)
      use iso_c_binding
      use array_descriptor
      type(gt_fortran_array_descriptor) :: arg0
      type(gt_fortran_array_descriptor) :: arg1
    end subroutine
    subroutine test_c_bindings_and_wrapper_compatible_type_b_impl(arg0, arg1) bind(c, &
        name="test_c_bindings_and_wrapper_compatible_type_b")
      use iso_c_binding
      use array_descriptor
      type(gt_fortran_array_descriptor) :: arg0
      type(gt_fortran_array_descriptor) :: arg1
    end subroutine

  end interface
  interface my_assign
    procedure my_assign0, my_assign1
  end interface
  interface my_push
    procedure my_push0, my_push1, my_push2
  end interface
contains
    subroutine my_assign0(arg0, arg1)
      use iso_c_binding
      use array_descriptor
      integer(c_int), dimension(:,:), target :: arg0
      integer(c_int), value, target :: arg1
      type(gt_fortran_array_descriptor) :: descriptor0

      descriptor0%rank = 2
      descriptor0%type = 1
      descriptor0%dims = reshape(shape(arg0), &
        shape(descriptor0%dims), (/0/))
      descriptor0%data = c_loc(arg0(lbound(arg0, 1),lbound(arg0, 2)))

      call my_assign0_impl(descriptor0, arg1)
    end subroutine
    subroutine my_assign1(arg0, arg1)
      use iso_c_binding
      use array_descriptor
      real(c_double), dimension(:,:), target :: arg0
      real(c_double), value, target :: arg1
      type(gt_fortran_array_descriptor) :: descriptor0

      descriptor0%rank = 2
      descriptor0%type = 6
      descriptor0%dims = reshape(shape(arg0), &
        shape(descriptor0%dims), (/0/))
      descriptor0%data = c_loc(arg0(lbound(arg0, 1),lbound(arg0, 2)))

      call my_assign1_impl(descriptor0, arg1)
    end subroutine
    subroutine test_c_bindings_and_wrapper_compatible_type_b(arg0, arg1)
      use iso_c_binding
      use array_descriptor
      type(gt_fortran_array_descriptor), target :: arg0
      integer(c_int), dimension(:,:), target :: arg1
      type(gt_fortran_array_descriptor) :: descriptor1

      descriptor1%rank = 2
      descriptor1%type = 1
      descriptor1%dims = reshape(shape(arg1), &
        shape(descriptor1%dims), (/0/))
      descriptor1%data = c_loc(arg1(lbound(arg1, 1),lbound(arg1, 2)))

      call test_c_bindings_and_wrapper_compatible_type_b_impl(arg0, descriptor1)
    end subroutine
end
)?";

    TEST(export, fortran_interface) {
        std::ostringstream strm;
        gridtools::c_bindings::generate_fortran_interface(strm, "my_module");
        EXPECT_EQ(strm.str(), expected_fortran_interface);
    }
} // namespace
