#pragma once

#ifndef __Size0
#define __Size0 12
#endif

#ifndef __Size1
#define __Size1 33
#endif

#ifndef __Size2
#define __Size2 61
#endif

TEST(TESTCLASS, copies3D) {
    EXPECT_EQ((copy_stencils_3D_2D_1D_0D::test<gridtools::layout_map<0,1,2> , gridtools::layout_map<0,1,2> >(__Size0, __Size1, __Size2)), true);
}

TEST(TESTCLASS, copies3Dtranspose) {
    EXPECT_EQ((copy_stencils_3D_2D_1D_0D::test<gridtools::layout_map<2,1,0> , gridtools::layout_map<0,1,2> >(__Size0, __Size1, __Size2)), true);
}

TEST(TESTCLASS, copies2Dij) {
    EXPECT_EQ((copy_stencils_3D_2D_1D_0D::test<gridtools::layout_map<0,1,-1> , gridtools::layout_map<0,1,2> >(__Size0, __Size1, __Size2)), true);
}

TEST(TESTCLASS, copies2Dik) {
    EXPECT_EQ((copy_stencils_3D_2D_1D_0D::test<gridtools::layout_map<0,-1,1> , gridtools::layout_map<0,1,2> >(__Size0, __Size1, __Size2)), true);
}

TEST(TESTCLASS, copies2Djk) {
    EXPECT_EQ((copy_stencils_3D_2D_1D_0D::test<gridtools::layout_map<-1,0,1> , gridtools::layout_map<0,1,2> >(__Size0, __Size1, __Size2)), true);
}

TEST(TESTCLASS, copies2Di) {
    EXPECT_EQ((copy_stencils_3D_2D_1D_0D::test<gridtools::layout_map<0,-1,-1> , gridtools::layout_map<0,1,2> >(__Size0, __Size1, __Size2)), true);
}

TEST(TESTCLASS, copies2Dj) {
    EXPECT_EQ((copy_stencils_3D_2D_1D_0D::test<gridtools::layout_map<-1,0,-1> , gridtools::layout_map<0,1,2> >(__Size0, __Size1, __Size2)), true);
}

TEST(TESTCLASS, copies2Dk) {
    EXPECT_EQ((copy_stencils_3D_2D_1D_0D::test<gridtools::layout_map<-1,-1,0> , gridtools::layout_map<0,1,2> >(__Size0, __Size1, __Size2)), true);
}

TEST(TESTCLASS, copies2DScalar) {
    EXPECT_EQ((copy_stencils_3D_2D_1D_0D::test<gridtools::layout_map<-1,-1,-1> , gridtools::layout_map<0,1,2> >(__Size0, __Size1, __Size2)), true);
}

TEST(TESTCLASS, copies3DDst) {
    EXPECT_EQ((copy_stencils_3D_2D_1D_0D::test<gridtools::layout_map<0,1,2> , gridtools::layout_map<2,0,1> >(__Size0, __Size1, __Size2)), true);
}

TEST(TESTCLASS, copies3DtransposeDst) {
    EXPECT_EQ((copy_stencils_3D_2D_1D_0D::test<gridtools::layout_map<2,1,0> , gridtools::layout_map<2,0,1> >(__Size0, __Size1, __Size2)), true);
}

TEST(TESTCLASS, copies2DijDst) {
    EXPECT_EQ((copy_stencils_3D_2D_1D_0D::test<gridtools::layout_map<1,0,-1> , gridtools::layout_map<2,0,1> >(__Size0, __Size1, __Size2)), true);
}

TEST(TESTCLASS, copies2DikDst) {
    EXPECT_EQ((copy_stencils_3D_2D_1D_0D::test<gridtools::layout_map<1,-1,0> , gridtools::layout_map<2,0,1> >(__Size0, __Size1, __Size2)), true);
}

TEST(TESTCLASS, copies2DjkDst) {
    EXPECT_EQ((copy_stencils_3D_2D_1D_0D::test<gridtools::layout_map<-1,1,0> , gridtools::layout_map<2,0,1> >(__Size0, __Size1, __Size2)), true);
}

TEST(TESTCLASS, copies2DiDst) {
    EXPECT_EQ((copy_stencils_3D_2D_1D_0D::test<gridtools::layout_map<0,-1,-1> , gridtools::layout_map<2,0,1> >(__Size0, __Size1, __Size2)), true);
}

TEST(TESTCLASS, copies2DjDst) {
    EXPECT_EQ((copy_stencils_3D_2D_1D_0D::test<gridtools::layout_map<-1,0,-1> , gridtools::layout_map<2,0,1> >(__Size0, __Size1, __Size2)), true);
}

TEST(TESTCLASS, copies2DkDst) {
    EXPECT_EQ((copy_stencils_3D_2D_1D_0D::test<gridtools::layout_map<-1,-1,0> , gridtools::layout_map<2,0,1> >(__Size0, __Size1, __Size2)), true);
}

TEST(TESTCLASS, copies2DScalarDst) {
    EXPECT_EQ((copy_stencils_3D_2D_1D_0D::test<gridtools::layout_map<-1,-1,-1> , gridtools::layout_map<2,0,1> >(__Size0, __Size1, __Size2)), true);
}
