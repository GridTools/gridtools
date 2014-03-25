

#include "gtest/gtest.h"

#include "gpu_clone.cu.h"
#include "cloningstuff.cu.h"
#include "test_domain.h"
#include "test_cuda_storage.h"
#include "test_hybrid_pointer.h"

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

TEST(testgpuclone, testcloningstuff) {
    EXPECT_EQ(cloningstuff_test::test_cloningstuff(), true);
}


int main(int argc, char** argv) {

    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
