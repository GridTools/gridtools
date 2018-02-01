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
    GT_EXPORT_BINDING_2(my_push, push_impl< double >);
    GT_EXPORT_GENERIC_BINDING(2, my_push_, push_impl, float);

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
        my_push(obj, 42);
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
void my_push(gt_handle*, double);
void my_push_0(gt_handle*, float);
double my_top(gt_handle*);

#ifdef __cplusplus
}
#endif
)?";

    TEST(export, c_interface) {
        std::ostringstream strm;
        EXPECT_EQ(gridtools::c_bindings::generate_c_interface(strm).str(), expected_c_interface);
    }
}
