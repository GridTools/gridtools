/*
   Copyright 2016 GridTools Consortium

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
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
