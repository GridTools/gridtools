#define CUDA_EXAMPLE

#include "gtest/gtest.h"

#define CUDA_EXAMPLE
#include "gpu_clone.cu.h"
#include "cloningstuff.cu.h"
#include "test_domain.h"
#include "test_cuda_storage.h"
#include "test_hybrid_pointer.h"
#include "../examples/interface1.h"
#include "copies_2D_1D_0D.h"
#include "../examples/tridiagonal.h"
#include "../examples/positional_copy_stencil.h"

#include "boundary_conditions_test.h"

TEST(testdomain, testallocationongpu) {
    EXPECT_EQ(test_domain(), false);
}

TEST(testhybridpointer, testhybridpointerongpu) {
    EXPECT_EQ(test_hybrid_pointer(), true);
}

TEST(testcudastorage, testcudastorageongpu) {
    EXPECT_EQ(test_cuda_storage(), true);
}

TEST(testgpuclone, testgpuclone) {
    EXPECT_EQ(gpu_clone_test::test_gpu_clone(), true);
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

TEST(testgpuclone, testcloningstuff) {
    EXPECT_EQ(cloningstuff_test::test_cloningstuff(), true);
}

TEST(stencil, horizontaldiffusion) {
    EXPECT_EQ(horizontal_diffusion::test(16, 16, 5), true);
}

#define __Size0 52
#define __Size1 52
#define __Size2 60

#define TESTCLASS stencil_cuda
#include "stencil_tests.h"
#undef TESTCLASS


int main(int argc, char** argv) {

    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
