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

#include "c_bindings/export.hpp"

#include <functional>
#include <sstream>
#include <stack>

#include <gtest/gtest.h>

#include "c_bindings/handle.hpp"

namespace {

    using stack_t = std::stack< double >;

    // Various flavours to create exported functions.

    stack_t my_create_impl() { return stack_t{}; }
    GT_EXPORT_BINDING_0(my_create, my_create_impl);

    template < class T >
    void push_impl(std::stack< T > &obj, T val) {
        obj.push(val);
    }
    GT_EXPORT_GENERIC_BINDING(2, my_push, push_impl, (float, int, double));

    GT_EXPORT_BINDING_WITH_SIGNATURE_1(my_pop, void(stack_t &), [](stack_t &obj) { obj.pop(); });

    struct top_impl {
        template < class T >
        T operator()(std::stack< T > const &container) const {
            return container.top();
        }
    };
    GT_EXPORT_BINDING_WITH_SIGNATURE_1(my_top, double(stack_t const &), top_impl{});

    GT_EXPORT_BINDING_WITH_SIGNATURE_1(my_empty, bool(stack_t const &), std::mem_fn(&stack_t::empty));

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

    const char expected_c_interface[] = R"?(
struct gt_handle;

#ifdef __cplusplus
extern "C" {
#else
typedef struct gt_handle gt_handle;
#endif

void gt_release(gt_handle*);
gt_handle* my_create();
bool my_empty(gt_handle*);
void my_pop(gt_handle*);
void my_push0(gt_handle*, float);
void my_push1(gt_handle*, int);
void my_push2(gt_handle*, double);
double my_top(gt_handle*);

#ifdef __cplusplus
}
#endif
)?";

    TEST(export, c_interface) {
        std::ostringstream strm;
        gridtools::c_bindings::generate_c_interface(strm);
        EXPECT_EQ(strm.str(), expected_c_interface);
    }

    const char expected_fortran_interface[] = R"?(
module gt_import
implicit none
  interface

    subroutine gt_release(h) bind(c)
      use iso_c_binding
      type(c_ptr), value :: h
    end
    type(c_ptr) function my_create() bind(c)
      use iso_c_binding
    end
    logical(c_bool) function my_empty(arg0) bind(c)
      use iso_c_binding
      type(c_ptr), value :: arg0
    end
    subroutine my_pop(arg0) bind(c)
      use iso_c_binding
      type(c_ptr), value :: arg0
    end
    subroutine my_push0(arg0, arg1) bind(c)
      use iso_c_binding
      type(c_ptr), value :: arg0
      real(c_float), value :: arg1
    end
    subroutine my_push1(arg0, arg1) bind(c)
      use iso_c_binding
      type(c_ptr), value :: arg0
      integer(c_int), value :: arg1
    end
    subroutine my_push2(arg0, arg1) bind(c)
      use iso_c_binding
      type(c_ptr), value :: arg0
      real(c_double), value :: arg1
    end
    real(c_double) function my_top(arg0) bind(c)
      use iso_c_binding
      type(c_ptr), value :: arg0
    end

  end interface
  interface my_push
    procedure my_push0, my_push1, my_push2
  end interface
end
)?";

    TEST(export, fortran_interface) {
        std::ostringstream strm;
        gridtools::c_bindings::generate_fortran_interface(strm);
        EXPECT_EQ(strm.str(), expected_fortran_interface);
    }
}
