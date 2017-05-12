/*
  GridTools Libraries

  Copyright (c) 2017, ETH Zurich and MeteoSwiss
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
/**
   The following class describes a boolean list of length N.
   This is used in proc_grids.

   It accepts an integer template arguemnt that is the length of the list
   and a sequence of boolean template arguments.

   \code
   boollist<...> bl(....);
   if (bl.value0) {
      ...
   }
   if (!bl.value2) {
      ...
   }
   \endcode
   See \link Concepts \endlink, \link proc_grid_2D_concept \endlink, \link proc_grid_3D_concept \endlink

   Additionally a boollist should provide a method to return a
   boollist with values permuted according to a \link
   gridtools::layout_map \endlink .

   This method has the following signature:
   \code
   boollist<N> B2 = B1.permute<gridtools::layout_map<I1,I2,I3> >();
   \endcode

   Where I1, I2, and I3 specify a permutation of the numbers from
   0 to N-1. Now B2.value0 is equal to B1.value<I> where I is the
   position of 0 in the layout_map; B2.value1 is equal to
   B1.value<I> where I is the position of 1 in the layout_map;
   B3.value0 is equal to B1.value<I> where I is the position of 3
   in the layout_map

   An implementation is found in struct gridtools::gcl_utils::boollist
 */
struct boollist_concept {}
