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

#include <c_bindings/handle.hpp>
#include <c_bindings/generator.hpp>

#include <sstream>

#include <gtest/gtest.h>

namespace gridtools {
    namespace c_bindings {
        namespace {

            GT_ADD_GENERATED_DECLARATION(void(), foo);
            GT_ADD_GENERATED_DECLARATION(gt_handle *(int, double const *, gt_handle *), bar);
            GT_ADD_GENERATED_DECLARATION(void(int *const *volatile *const *), baz);

            GT_ADD_GENERIC_DECLARATION(foo, bar);
            GT_ADD_GENERIC_DECLARATION(foo, baz);

            const char expected_c_interface[] = R"?(
struct gt_handle;

#ifdef __cplusplus
extern "C" {
#else
typedef struct gt_handle gt_handle;
#endif

void gt_release(gt_handle*);
gt_handle* bar(int, double*, gt_handle*);
void baz(int****);
void foo();

#ifdef __cplusplus
}
#endif
)?";

            TEST(generator, c_interface) {
                std::ostringstream strm;
                generate_c_interface(strm);
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
    type(c_ptr) function bar(arg0, arg1, arg2) bind(c)
      use iso_c_binding
      integer(c_int), value :: arg0
      real(c_double), dimension(*) :: arg1
      type(c_ptr), value :: arg2
    end
    subroutine baz(arg0) bind(c)
      use iso_c_binding
      type(c_ptr) :: arg0
    end
    subroutine foo() bind(c)
      use iso_c_binding
    end

  end interface
  interface foo
    procedure bar, baz
  end interface
end
)?";

            TEST(generator, fortran_interface) {
                std::ostringstream strm;
                generate_fortran_interface(strm);
                EXPECT_EQ(strm.str(), expected_fortran_interface);
            }
        }
    }
}
