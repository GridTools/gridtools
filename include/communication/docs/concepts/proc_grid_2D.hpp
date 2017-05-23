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

/** class proc_grid_2D_concept
 This is not a real class but only a template to illustrate the concept of a 2D process grid
 Given a contiguos range of P processes, from 0 to P-1, this class provide a distribution of
 these processes as a 2D grid by row.

 \tparam PERIOD Type of the object providing periodicity information
 */
template < typename PERIOD >
struct proc_grid_2D_concept {

    /** Type of the object to handle periodicities
     */
    typedef... period_type

        /** number of dimensions
         */
        static const int ndims = 2;

    /** Returns in t_R and t_C the lenght of the dimensions of the process grid
        \param[out] t_R Number of elements in first dimension
        \param[out] t_C Number of elements in second dimension
     */
    void dims(int &t_R, int &t_C) const { ... }

    /** Returns the number of processors of the processor grid

        \return Number of processors
    */
    int size() const { ... }

    /** Returns in t_R and t_C the coordinates ot the caller process in the grid
        \param[out] t_R Coordinate in first dimension
        \param[out] t_C Coordinate in second dimension
     */
    void coords(int &t_R, int &t_C) const { ... }

    /** Returns the process ID of the process with relative coordinates (I,J) with respect to the caller process
        \tparam I Relative coordinate in the first dimension
        \tparam J Relative coordinate in the seocnd dimension
        \return The process ID of the required process
     */
    template < int I, int J >
    int proc() const {
        ...
    }

    /** Returns the process ID of the process with relative coordinates (I,J) with respect to the caller process
        \param[in] I Relative coordinate in the first dimension
        \param[in] J Relative coordinate in the seocnd dimension
        \return The process ID of the required process
     */
    int proc(int I, int J) const { ... }

    /** Returns the process ID of the process with absolute coordinates specified by the input gridtools::array of
       coordinates
        \param[in] crds gridtools::aray of coordinates of the processor of which the ID is needed

        \return The process ID of the required process
    */
    int abs_proc(gridtools::array< int, ndims > const &crds) const { ... }
};
