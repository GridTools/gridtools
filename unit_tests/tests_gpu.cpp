#define CUDA_EXAMPLE

#include "gtest/gtest.h"

#define CUDA_EXAMPLE
#include "gpu_clone.cu.hpp"
#include "cloningstuff.cu.hpp"
#include "test_domain.hpp"
#include "test_cuda_storage.hpp"
#include "test_hybrid_pointer.hpp"
#include "../examples/interface1.hpp"
#include "copies_2D_1D_0D.hpp"
#include "../examples/tridiagonal.hpp"
#include "../examples/positional_copy_stencil.hpp"

#include "boundary_conditions_test.hpp"

//TODO modify after the memory leak fix
// TEST(testdomain, testallocationongpu) {
//     EXPECT_EQ(test_domain(), false);
// }

TEST(testhybridpointer, testhybridpointerongpu) {
    EXPECT_EQ(test_hybrid_pointer(), true);
}

// TEST(testcudastorage, testcudastorageongpu) {
//     EXPECT_EQ(test_cuda_storage(), true);
// }

TEST(testgpuclone, testgpuclone) {
    EXPECT_EQ(gpu_clone_test::test_gpu_clone(), true);
}

// //the access to the storage metadata map is to be implemented for the boundary conditions
// TEST(boundaryconditions, basic) {
//     EXPECT_EQ(basic(), true);
// }

// TEST(boundaryconditions, predicate) {
//     EXPECT_EQ(predicate(), true);
// }

// TEST(boundaryconditions, twosurfaces) {
//     EXPECT_EQ(twosurfaces(), true);
// }

// TEST(boundaryconditions, usingzero1) {
//     EXPECT_EQ(usingzero_1(), true);
// }

// TEST(boundaryconditions, usingzero2) {
//     EXPECT_EQ(usingzero_2(), true);
// }

// TEST(boundaryconditions, usingvalue2) {
//     EXPECT_EQ(usingvalue_2(), true);
// }

// TEST(boundaryconditions, usingcopy3) {
//     EXPECT_EQ(usingcopy_3(), true);
// }

TEST(testgpuclone, testcloningstuff) {
    EXPECT_EQ(cloningstuff_test::test_cloningstuff(), true);
}

#define __Size0 52
#define __Size1 52
#define __Size2 60

#define TESTCLASS stencil_cuda
#include "stencil_tests.hpp"
#undef TESTCLASS


int main(int argc, char** argv) {

    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
