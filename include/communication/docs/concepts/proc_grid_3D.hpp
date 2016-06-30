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

/** class proc_grid_3D_concept
 * This is not a real class but only a template to illustrate the concept of a 2D process grid
 * Given a contiguos range of P processes, from 0 to P-1, this class provide a distribution of
 * these processes as a 2D grid by row. For example, processes 0,1,2,3,4,5 will be layed out as

 \tparam PERIOD Type of the object providing periodicity information
 */
template < typename PERIOD >
struct proc_grid_3D_concept {

    /** Type of the object to handle periodicities
     */
    typedef... period_type

        /** number of dimensions
         */
        static const int ndims = 2;

    /** Returns in t_R and t_C the lenght of the dimensions of the process grid
        \param[out] t_R Number of elements in first dimension
        \param[out] t_C Number of elements in second dimension
        \param[out] t_S Number of elements in third dimension
     */
    void dims(int &t_R, int &t_C, int &t_S) const { ... }

    /** Returns in t_R and t_C the coordinates ot the caller process in the grid
        \param[out] t_R Coordinate in first dimension
        \param[out] t_C Coordinate in second dimension
        \param[out] t_S Coordinate in third dimension
     */
    void coords(int &t_R, int &t_C, int &t_S) const { ... }

    /** Returns the process ID of the process with relative coordinates (I,J) with respect to the caller process
        \tparam I Relative coordinate in the first dimension
        \tparam J Relative coordinate in the seocnd dimension
        \tparam K Relative coordinate in the third dimension
        \return The process ID of the required process
     */
    template < int I, int J, int K >
    int proc() const {
        ...
    }

    /** Returns the process ID of the process with relative coordinates (I,J) with respect to the caller process
        \param[in] I Relative coordinate in the first dimension
        \param[in] J Relative coordinate in the seocnd dimension
        \param[in] K Relative coordinate in the third dimension
        \return The process ID of the required process
     */
    int proc(int I, int J, int K) const { ... }

    /** Returns the process ID of the process with absolute coordinates specified by the input gridtools::array of
       coordinates
        \param[in] crds gridtools::aray of coordinates of the processor of which the ID is needed

        \return The process ID of the required process
    */
    int abs_proc(gridtools::array< int, ndims > const &crds) const { ... }
};
