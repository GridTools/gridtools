#include "gtest/gtest.h"

#include "test_domain_indices.h"
#include "test_smallstorage_indices.h"
#include "boundary_conditions_test.h"

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


int main(int argc, char** argv) {

    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
