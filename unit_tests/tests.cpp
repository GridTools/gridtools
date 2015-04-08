#include "gtest/gtest.h"

#define SILENT_RUN
#include "test_domain_indices.h"
#include "test_smallstorage_indices.h"
#include "boundary_conditions_test.h"
#include "../examples/interface1.h"
#include "../examples/positional_copy_stencil.h"
#include "copies_2D_1D_0D.h"
#include "../examples/tridiagonal.h"
#include "../examples/extended_4D.h"
#include "external_ptr_test/CopyStencil.h"
#ifdef CXX11_ENABLED
#include "test-assign-placeholders.h"
#endif
#include "arg_type_tests.h"

#include "communication/layout_map.cpp"

TEST(testdomain, testindices) {
    EXPECT_EQ(test_domain_indices(), true);
}

TEST(testsmallstorage, testindices) {
    EXPECT_EQ(test_smallstorage_indices(), true);
}

TEST(interface, arg_type1) {
    EXPECT_EQ(interface::test_trivial(), true);
}
TEST(interface, arg_type2) {
    EXPECT_EQ(interface::test_alternative1(), true);
}

#ifdef CXX11_ENABLED

TEST(interface, arg_type3) {
    EXPECT_EQ(interface::test_alternative2(), true);
}
TEST(interface, arg_type4) {
    EXPECT_EQ(interface::test_static_alias(), true);
}
TEST(interface, arg_type5) {
    EXPECT_EQ(interface::test_dynamic_alias(), true);
}
#endif

#define BACKEND_BLOCK
#define TESTCLASS stencil_block
#include "stencil_tests.h"
#undef BACKEND_BLOCK
#undef TESTCLASS
#define TESTCLASS stencil
#include "stencil_tests.h"

#ifdef CXX11_ENABLED
TEST(stencil, extended_4D) {
    EXPECT_EQ(assembly::test(5, 5, 6), true);
}
#endif

#ifdef CXX11_ENABLED
TEST(testdomain, assignplchdrs) {
    EXPECT_EQ(assign_placeholders(), true);
}
#endif

TEST(testcommon, layoutmap) {
    EXPECT_EQ(test_layout_map(), true);
}

TEST(python, copy) {
    EXPECT_EQ(test_copystencil_python(), false);
}

int main(int argc, char** argv)
{

    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
