/*
  GridTools Libraries

  Copyright (c) 2016, GridTools Consortium
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

#define SILENT_RUN
#include "test_domain_indices.hpp"
#include "boundary_conditions_test.hpp"
#include "copies_2D_1D_0D.hpp"
#include "external_ptr_test/CopyStencil.hpp"
#include "accessor_tests.hpp"
#include "loop_hierarchy_test.hpp"

TEST(testdomain, testindices) {
    EXPECT_EQ(test_domain_indices(), true);
}

TEST(interface, accessor0) {
    EXPECT_EQ(interface::test_trivial(), true);
}
TEST(interface, accessor1) {
    EXPECT_EQ(interface::test_alternative1(), true);
}

#ifdef CXX11_ENABLED

TEST(interface, accessor2) {
    EXPECT_EQ(interface::test_alternative2(), true);
}
TEST(interface, accessor3) {
    EXPECT_EQ(interface::test_static_alias(), true);
}
TEST(interface, accessor4) {
    EXPECT_EQ(interface::test_dynamic_alias(), true);
}
#endif


TEST(boundaryconditions, usingvalue2) {
    EXPECT_EQ(usingvalue_2(), true);
}

TEST(boundaryconditions, usingcopy3) {
    EXPECT_EQ(usingcopy_3(), true);
}

TEST(stencil, loop_hierarchy) {
    EXPECT_EQ(loop_test::test(), true);
}

#define BACKEND_BLOCK
#define TESTCLASS stencil_block
#include "stencil_tests.hpp"
#undef BACKEND_BLOCK
#undef TESTCLASS
#define TESTCLASS stencil
#include "stencil_tests.hpp"

TEST(python, copy) {
    EXPECT_EQ(test_copystencil_python(), false);
}

int main(int argc, char** argv)
{

    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
