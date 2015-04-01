#include "gtest/gtest.h"

#define SILENT_RUN
#include "test_domain_indices.h"
#include "test_smallstorage_indices.h"
//#include "boundary_conditions_test.h"
#include "../examples/interface1.h"
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

TEST(stencil, horizontaldiffusion) {
    EXPECT_EQ(horizontal_diffusion::test(7, 13, 5), true);
}

#define BACKEND_BLOCK
TEST(stencil, horizontaldiffusion_block) {
    EXPECT_EQ(horizontal_diffusion::test(7, 13, 5), true);
}

#undef BACKEND_BLOCK

#define __Size0 5
#define __Size1 8
#define __Size2 6

TEST(stencil, copies3D) {
    EXPECT_EQ((copy_stencils_3D_2D_1D_0D::test<gridtools::layout_map<0,1,2> , gridtools::layout_map<0,1,2> >(__Size0, __Size1, __Size2)), true);
}

TEST(stencil, copies3Dtranspose) {
    EXPECT_EQ((copy_stencils_3D_2D_1D_0D::test<gridtools::layout_map<2,1,0> , gridtools::layout_map<0,1,2> >(__Size0, __Size1, __Size2)), true);
}

TEST(stencil, copies2Dij) {
    EXPECT_EQ((copy_stencils_3D_2D_1D_0D::test<gridtools::layout_map<0,1,-1> , gridtools::layout_map<0,1,2> >(__Size0, __Size1, __Size2)), true);
}

TEST(stencil, copies2Dik) {
    EXPECT_EQ((copy_stencils_3D_2D_1D_0D::test<gridtools::layout_map<0,-1,1> , gridtools::layout_map<0,1,2> >(__Size0, __Size1, __Size2)), true);
}

TEST(stencil, copies2Djk) {
    EXPECT_EQ((copy_stencils_3D_2D_1D_0D::test<gridtools::layout_map<-1,0,1> , gridtools::layout_map<0,1,2> >(__Size0, __Size1, __Size2)), true);
}

TEST(stencil, copies2Di) {
    EXPECT_EQ((copy_stencils_3D_2D_1D_0D::test<gridtools::layout_map<0,-1,-1> , gridtools::layout_map<0,1,2> >(__Size0, __Size1, __Size2)), true);
}

TEST(stencil, copies2Dj) {
    EXPECT_EQ((copy_stencils_3D_2D_1D_0D::test<gridtools::layout_map<-1,0,-1> , gridtools::layout_map<0,1,2> >(__Size0, __Size1, __Size2)), true);
}

TEST(stencil, copies2Dk) {
    EXPECT_EQ((copy_stencils_3D_2D_1D_0D::test<gridtools::layout_map<-1,-1,0> , gridtools::layout_map<0,1,2> >(__Size0, __Size1, __Size2)), true);
}

TEST(stencil, copies2DScalar) {
    EXPECT_EQ((copy_stencils_3D_2D_1D_0D::test<gridtools::layout_map<-1,-1,-1> , gridtools::layout_map<0,1,2> >(__Size0, __Size1, __Size2)), true);
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

TEST(stencil, extended_4D) {
    EXPECT_EQ(assembly::test(5, 5, 6), true);
}

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
