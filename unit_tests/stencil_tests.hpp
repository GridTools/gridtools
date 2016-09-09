/*
  GridTools Libraries

  Copyright (c) 2016, GridTools Consortium
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  1. Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

  For information: http://eth-cscs.github.io/gridtools/
*/
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
    EXPECT_EQ((copy_stencils_3D_2D_1D_0D::test< gridtools::layout_map< 0, 1, 2 >, gridtools::layout_map< 0, 1, 2 > >(
                  __Size0, __Size1, __Size2)),
        true);
}

TEST(TESTCLASS, copies3Dtranspose) {
    EXPECT_EQ((copy_stencils_3D_2D_1D_0D::test< gridtools::layout_map< 2, 1, 0 >, gridtools::layout_map< 0, 1, 2 > >(
                  __Size0, __Size1, __Size2)),
        true);
}

TEST(TESTCLASS, copies2Dij) {
    EXPECT_EQ((copy_stencils_3D_2D_1D_0D::test< gridtools::layout_map< 0, 1, -1 >, gridtools::layout_map< 0, 1, 2 > >(
                  __Size0, __Size1, __Size2)),
        true);
}

TEST(TESTCLASS, copies2Dik) {
    EXPECT_EQ((copy_stencils_3D_2D_1D_0D::test< gridtools::layout_map< 0, -1, 1 >, gridtools::layout_map< 0, 1, 2 > >(
                  __Size0, __Size1, __Size2)),
        true);
}

TEST(TESTCLASS, copies2Djk) {
    EXPECT_EQ((copy_stencils_3D_2D_1D_0D::test< gridtools::layout_map< -1, 0, 1 >, gridtools::layout_map< 0, 1, 2 > >(
                  __Size0, __Size1, __Size2)),
        true);
}

TEST(TESTCLASS, copies2Di) {
    EXPECT_EQ((copy_stencils_3D_2D_1D_0D::test< gridtools::layout_map< 0, -1, -1 >, gridtools::layout_map< 0, 1, 2 > >(
                  __Size0, __Size1, __Size2)),
        true);
}

TEST(TESTCLASS, copies2Dj) {
    EXPECT_EQ((copy_stencils_3D_2D_1D_0D::test< gridtools::layout_map< -1, 0, -1 >, gridtools::layout_map< 0, 1, 2 > >(
                  __Size0, __Size1, __Size2)),
        true);
}

TEST(TESTCLASS, copies2Dk) {
    EXPECT_EQ((copy_stencils_3D_2D_1D_0D::test< gridtools::layout_map< -1, -1, 0 >, gridtools::layout_map< 0, 1, 2 > >(
                  __Size0, __Size1, __Size2)),
        true);
}

TEST(TESTCLASS, copies2DScalar) {
    EXPECT_EQ((copy_stencils_3D_2D_1D_0D::test< gridtools::layout_map< -1, -1, -1 >, gridtools::layout_map< 0, 1, 2 > >(
                  __Size0, __Size1, __Size2)),
        true);
}

TEST(TESTCLASS, copies3DDst) {
    EXPECT_EQ((copy_stencils_3D_2D_1D_0D::test< gridtools::layout_map< 0, 1, 2 >, gridtools::layout_map< 2, 0, 1 > >(
                  __Size0, __Size1, __Size2)),
        true);
}

TEST(TESTCLASS, copies3DtransposeDst) {
    EXPECT_EQ((copy_stencils_3D_2D_1D_0D::test< gridtools::layout_map< 2, 1, 0 >, gridtools::layout_map< 2, 0, 1 > >(
                  __Size0, __Size1, __Size2)),
        true);
}

TEST(TESTCLASS, copies2DijDst) {
    EXPECT_EQ((copy_stencils_3D_2D_1D_0D::test< gridtools::layout_map< 1, 0, -1 >, gridtools::layout_map< 2, 0, 1 > >(
                  __Size0, __Size1, __Size2)),
        true);
}

TEST(TESTCLASS, copies2DikDst) {
    EXPECT_EQ((copy_stencils_3D_2D_1D_0D::test< gridtools::layout_map< 1, -1, 0 >, gridtools::layout_map< 2, 0, 1 > >(
                  __Size0, __Size1, __Size2)),
        true);
}

TEST(TESTCLASS, copies2DjkDst) {
    EXPECT_EQ((copy_stencils_3D_2D_1D_0D::test< gridtools::layout_map< -1, 1, 0 >, gridtools::layout_map< 2, 0, 1 > >(
                  __Size0, __Size1, __Size2)),
        true);
}

TEST(TESTCLASS, copies2DiDst) {
    EXPECT_EQ((copy_stencils_3D_2D_1D_0D::test< gridtools::layout_map< 0, -1, -1 >, gridtools::layout_map< 2, 0, 1 > >(
                  __Size0, __Size1, __Size2)),
        true);
}

TEST(TESTCLASS, copies2DjDst) {
    EXPECT_EQ((copy_stencils_3D_2D_1D_0D::test< gridtools::layout_map< -1, 0, -1 >, gridtools::layout_map< 2, 0, 1 > >(
                  __Size0, __Size1, __Size2)),
        true);
}

TEST(TESTCLASS, copies2DkDst) {
    EXPECT_EQ((copy_stencils_3D_2D_1D_0D::test< gridtools::layout_map< -1, -1, 0 >, gridtools::layout_map< 2, 0, 1 > >(
                  __Size0, __Size1, __Size2)),
        true);
}

TEST(TESTCLASS, copies2DScalarDst) {
    EXPECT_EQ((copy_stencils_3D_2D_1D_0D::test< gridtools::layout_map< -1, -1, -1 >, gridtools::layout_map< 2, 0, 1 > >(
                  __Size0, __Size1, __Size2)),
        true);
}
