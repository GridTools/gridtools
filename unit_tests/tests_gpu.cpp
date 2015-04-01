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

#include "boundary_conditions_test.h"
#include "arg_type_tests.h"

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

#define __Size0 32
#define __Size1 32
#define __Size2 2

TEST(stencil, copies3D) {
    EXPECT_EQ((copy_stencils_3D_2D_1D_0D::test<gridtools::layout_map<2,1,0> , gridtools::layout_map<2,1,0> >(__Size0, __Size1, __Size2)), true);
}

TEST(stencil, copies3Dtranspose) {
    EXPECT_EQ((copy_stencils_3D_2D_1D_0D::test<gridtools::layout_map<0,1,2> , gridtools::layout_map<0,1,2> >(__Size0, __Size1, __Size2)), true);
}

TEST(stencil, copies2Dij) {
    EXPECT_EQ((copy_stencils_3D_2D_1D_0D::test<gridtools::layout_map<0,1,-1> , gridtools::layout_map<2,1,0> >(__Size0, __Size1, __Size2)), true);
}

TEST(stencil, copies2Dik) {
    EXPECT_EQ((copy_stencils_3D_2D_1D_0D::test<gridtools::layout_map<0,-1,1> , gridtools::layout_map<2,1,0> >(__Size0, __Size1, __Size2)), true);
}

TEST(stencil, copies2Djk) {
    EXPECT_EQ((copy_stencils_3D_2D_1D_0D::test<gridtools::layout_map<-1,0,1> , gridtools::layout_map<2,1,0> >(__Size0, __Size1, __Size2)), true);
}

TEST(stencil, copies2Di) {
    EXPECT_EQ((copy_stencils_3D_2D_1D_0D::test<gridtools::layout_map<0,-1,-1> , gridtools::layout_map<2,1,0> >(__Size0, __Size1, __Size2)), true);
}

TEST(stencil, copies2Dj) {
    EXPECT_EQ((copy_stencils_3D_2D_1D_0D::test<gridtools::layout_map<-1,0,-1> , gridtools::layout_map<2,1,0> >(__Size0, __Size1, __Size2)), true);
}

TEST(stencil, copies2Dk) {
    EXPECT_EQ((copy_stencils_3D_2D_1D_0D::test<gridtools::layout_map<-1,-1,0> , gridtools::layout_map<2,1,0> >(__Size0, __Size1, __Size2)), true);
}

TEST(stencil, copies2DScalar) {
    EXPECT_EQ((copy_stencils_3D_2D_1D_0D::test<gridtools::layout_map<-1,-1,-1> , gridtools::layout_map<2,1,0> >(__Size0, __Size1, __Size2)), true);
}

TEST(stencil, copies3DDst) {
    EXPECT_EQ((copy_stencils_3D_2D_1D_0D::test<gridtools::layout_map<0,1,2> , gridtools::layout_map<2,0,1> >(__Size0, __Size1, __Size2)), true);
}

TEST(stencil, copies3DtransposeDst) {
    EXPECT_EQ((copy_stencils_3D_2D_1D_0D::test<gridtools::layout_map<2,1,0> , gridtools::layout_map<2,0,1> >(__Size0, __Size1, __Size2)), true);
}

TEST(stencil, copies2DijDst) {
    EXPECT_EQ((copy_stencils_3D_2D_1D_0D::test<gridtools::layout_map<1,0,-1> , gridtools::layout_map<2,0,1> >(__Size0, __Size1, __Size2)), true);
}

TEST(stencil, copies2DikDst) {
    EXPECT_EQ((copy_stencils_3D_2D_1D_0D::test<gridtools::layout_map<1,-1,0> , gridtools::layout_map<2,0,1> >(__Size0, __Size1, __Size2)), true);
}

TEST(stencil, copies2DjkDst) {
    EXPECT_EQ((copy_stencils_3D_2D_1D_0D::test<gridtools::layout_map<-1,1,0> , gridtools::layout_map<2,0,1> >(__Size0, __Size1, __Size2)), true);
}

TEST(stencil, copies2DiDst) {
    EXPECT_EQ((copy_stencils_3D_2D_1D_0D::test<gridtools::layout_map<0,-1,-1> , gridtools::layout_map<2,0,1> >(__Size0, __Size1, __Size2)), true);
}

TEST(stencil, copies2DjDst) {
    EXPECT_EQ((copy_stencils_3D_2D_1D_0D::test<gridtools::layout_map<-1,0,-1> , gridtools::layout_map<2,0,1> >(__Size0, __Size1, __Size2)), true);
}

TEST(stencil, copies2DkDst) {
    EXPECT_EQ((copy_stencils_3D_2D_1D_0D::test<gridtools::layout_map<-1,-1,0> , gridtools::layout_map<2,0,1> >(__Size0, __Size1, __Size2)), true);
}

TEST(stencil, copies2DScalarDst) {
    EXPECT_EQ((copy_stencils_3D_2D_1D_0D::test<gridtools::layout_map<-1,-1,-1> , gridtools::layout_map<2,0,1> >(__Size0, __Size1, __Size2)), true);
}

TEST(stencil, tridiagonal) {
    EXPECT_EQ(tridiagonal::solver(1, 1, 6), true);
}

int main(int argc, char** argv) {

    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
