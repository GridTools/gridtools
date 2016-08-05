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

//the access to the storage metadata map is to be implemented for the boundary conditions
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
