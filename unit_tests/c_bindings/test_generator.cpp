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

#include <gridtools/c_bindings/generator.hpp>
#include <gridtools/c_bindings/handle_impl.hpp>

#include <sstream>

#include <gtest/gtest.h>

namespace gridtools {
    namespace c_bindings {
        namespace {

            GT_ADD_GENERATED_DECLARATION(void(), foo);
            GT_ADD_GENERATED_DECLARATION(gt_handle *(int, double const *, gt_handle *), bar);
            GT_ADD_GENERATED_DECLARATION(void(int *const *volatile *const *), baz);
            GT_ADD_GENERATED_DECLARATION_WRAPPED(void(int, int (&)[1][2][3]), qux);

            GT_ADD_GENERIC_DECLARATION(foo, bar);
            GT_ADD_GENERIC_DECLARATION(foo, baz);

            const char expected_c_interface[] = R"?(// This file is generated!
#pragma once

#include <gridtools/c_bindings/array_descriptor.h>
#include <gridtools/c_bindings/handle.h>

#ifdef __cplusplus
extern "C" {
#endif

gt_handle* bar(int, double*, gt_handle*);
void baz(int****);
void foo();
void qux(int, gt_fortran_array_descriptor*);

#ifdef __cplusplus
}
#endif
)?";

            TEST(generator, c_interface) {
                std::ostringstream strm;
                generate_c_interface(strm);
                EXPECT_EQ(strm.str(), expected_c_interface);
            }

            const char expected_fortran_interface[] = R"?(! This file is generated!
module my_module
implicit none
  interface

    type(c_ptr) function bar(arg0, arg1, arg2) bind(c)
      use iso_c_binding
      integer(c_int), value :: arg0
      real(c_double), dimension(*) :: arg1
      type(c_ptr), value :: arg2
    end function
    subroutine baz(arg0) bind(c)
      use iso_c_binding
      type(c_ptr) :: arg0
    end subroutine
    subroutine foo() bind(c)
      use iso_c_binding
    end subroutine
    subroutine qux_impl(arg0, arg1) bind(c, name="qux")
      use iso_c_binding
      use array_descriptor
      integer(c_int), value :: arg0
      type(gt_fortran_array_descriptor) :: arg1
    end subroutine

  end interface
  interface foo
    procedure bar, baz
  end interface
contains
    subroutine qux(arg0, arg1)
      use iso_c_binding
      use array_descriptor
      integer(c_int), value, target :: arg0
      integer(c_int), dimension(:,:,:), target :: arg1
      type(gt_fortran_array_descriptor) :: descriptor1

      descriptor1%rank = 3
      descriptor1%type = 1
      descriptor1%dims = reshape(shape(arg1), &
        shape(descriptor1%dims), (/0/))
      descriptor1%data = c_loc(arg1(lbound(arg1, 1),lbound(arg1, 2),lbound(arg1, 3)))

      call qux_impl(arg0, descriptor1)
    end subroutine
end
)?";

            TEST(generator, fortran_interface) {
                std::ostringstream strm;
                generate_fortran_interface(strm, "my_module");
                EXPECT_EQ(strm.str(), expected_fortran_interface);
            }
            TEST(generator, wrap_short_line) {
                const std::string prefix = "    ";
                const std::string line = "short line, short line";
                EXPECT_EQ(prefix + line + '\n', wrap_line(line, prefix));
            }
            TEST(generator, wrap_almost_full_line) {
                const std::string prefix = "    ";
                const std::string line = std::string(64, 'x') + "," + std::string(63, 'x');
                EXPECT_EQ(prefix + line + '\n', wrap_line(line, prefix));
            }
            TEST(generator, wrap_full_line) {
                const std::string prefix = "    ";
                const std::string line = std::string(64, 'x') + "," + std::string(64, 'x');
                EXPECT_EQ(prefix + std::string(64, 'x') + ", &" + '\n' + prefix + "   " + std::string(64, 'x') + "\n",
                    wrap_line(line, prefix));
            }
            TEST(generator, wrap_multiple_lines) {
                const std::string prefix = "    ";
                const std::string line = std::string(50, 'x') + "," + std::string(50, 'x') + "," +
                                         std::string(60, 'x') + "," + std::string(61, 'x') + "," +
                                         std::string(60, 'x') + "," + std::string(62, 'x') + "," +
                                         std::string(59, 'x') + "," + std::string(122, 'x');

                const std::string line1 = prefix + std::string(50, 'x') + "," + std::string(50, 'x') + ", &" + '\n';
                const std::string line2 =
                    prefix + "   " + std::string(60, 'x') + "," + std::string(61, 'x') + ", &" + '\n';
                const std::string line3 = prefix + "   " + std::string(60, 'x') + ", &" + '\n';
                const std::string line4 =
                    prefix + "   " + std::string(62, 'x') + "," + std::string(59, 'x') + ", &" + '\n';
                const std::string line5 = prefix + "   " + std::string(122, 'x') + '\n';

                EXPECT_EQ(line1 + line2 + line3 + line4 + line5, wrap_line(line, prefix));
            }
        } // namespace
    }     // namespace c_bindings
} // namespace gridtools
