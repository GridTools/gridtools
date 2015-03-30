#include "gtest/gtest.h"

#define SILENT_RUN
#include "test_domain_indices.h"
#include "test_smallstorage_indices.h"
#include "boundary_conditions_test.h"
#include "../examples/interface1.h"
#include "../examples/positional_copy_stencil.h"
#include "copies_2D_1D_0D.h"
#include "../examples/tridiagonal.h"
#ifdef CXX11_ENABLED
#include "test-assign-placeholders.h"
#endif

#include "communication/layout_map.cpp"

TEST(testdomain, testindices) {
    EXPECT_EQ(test_domain_indices(), true);
}

TEST(testsmallstorage, testindices) {
    EXPECT_EQ(test_smallstorage_indices(), true);
}

TEST(boundaryconditions, basic) {
    EXPECT_EQ(basic(), true);
}

TEST(boundaryconditions, predicate) {
    EXPECT_EQ(predicate(), true);
}

TEST(boundaryconditions, twosurfaces) {
    EXPECT_EQ(twosurfaces(), true);
}

TEST(boundaryconditions, usingzero1) {
    EXPECT_EQ(usingzero_1(), true);
}

TEST(boundaryconditions, usingzero2) {
    EXPECT_EQ(usingzero_2(), true);
}

TEST(boundaryconditions, usingvalue2) {
    EXPECT_EQ(usingvalue_2(), true);
}

TEST(boundaryconditions, usingcopy3) {
    EXPECT_EQ(usingcopy_3(), true);
}

#define BACKEND_BLOCK
#define TESTCLASS stencil_block
#include "stencil_tests.h"
#undef BACKEND_BLOCK
#undef TESTCLASS
#define TESTCLASS stencil
#include "stencil_tests.h"

#ifdef CXX11_ENABLED
TEST(testdomain, assignplchdrs) {
    EXPECT_EQ(assign_placeholders(), true);
}
#endif

TEST(testcommon, layoutmap) {
    EXPECT_EQ(test_layout_map(), true);
}

int main(int argc, char** argv)
{

    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
