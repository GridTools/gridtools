/*
   Copyright 2016 GridTools Consortium

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
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
