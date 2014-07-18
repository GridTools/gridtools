
/*
Copyright (c) 2012, MAURO BIANCO, UGO VARETTO, SWISS NATIONAL SUPERCOMPUTING CENTRE (CSCS)
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the Swiss National Supercomputing Centre (CSCS) nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL MAURO BIANCO, UGO VARETTO, OR 
SWISS NATIONAL SUPERCOMPUTING CENTRE (CSCS), BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/


#ifndef _PROC_GRIDS_2D_H_
#define _PROC_GRIDS_2D_H_

#include <iostream>
#include <cmath>
#include <GCL.h>
#include <utils/array.h>
#include <boost/type_traits/integral_constant.hpp>

// This file needs to be changed

/** \namespace GCL
 * All library classes, functions, and objects will reside in this namespace.
 */
namespace GCL {

  /** \class _2D_process_grid_t
   Class that provides a generic 2D process grid from a linear
   process distribution.  Given a contiguos range of P processes,
   from 0 to P-1, this class provide a distribution of these
   processes as a 2D grid by row in the best possible aspect
   ratio. For example, processes 0,1,2,3,4,5 will be layed out as
   \code
   0 1
   2 3
   4 5
   \endcode
   \n
   \tparam CYCLIC is a boollist class \link boollist_concept \endlink template parameter that specifies which dimensions of the grid are clyclic 
   \n
   This is a process grid matching the \ref proc_grid_concept concept
   */
  template <typename CYCLIC>
  struct _2D_process_grid_t {

  public:
    typedef boost::false_type has_communicator;
    typedef CYCLIC period_type;

    /** number of dimensions
     */
    static const int ndims = 2;

    period_type cyclic;
    int R, C;
    int r,c;

    /** Constructor that takes the number of processes and the caller ID to produce the grid
        \param[in] c The object of the class used to specify periodicity in each dimension as defined in \ref boollist_concept
        \param[in] P Number of processes that will make the grid
        \param[in] pid Number of processes that will make the grid
     */
    explicit _2D_process_grid_t(period_type const &c, int P, int pid)
      : cyclic(c)
      , R(0)
      , C(0)
      , r(0)
      , c(0) 
    {
      create(P,pid);
    }

    /** Function to create the grid. This can be called in case the grid is default constructed. Its direct use is discouraged
        \param[in] P Number of processes that will make the grid
        \param[in] pid Number of processes that will make the grid
     */
    void create(int P, int pid) {
      int sq = static_cast<int>(sqrt(static_cast<double>(P)));
      for (int i=1; i<=sq; ++i) {
        int t = P/i;
        if (i*t == P) {
          C = i;
          R = t;
        }
      }
      int my_pid = pid;
      r = my_pid/C;
      c = my_pid%C;
    }

    /** Returns in t_R and t_C the lenght of the dimensions of the process grid AS PRESCRIBED BY THE CONCEPT
        \param[out] t_R Number of elements in first dimension
        \param[out] t_C Number of elements in second dimension
    */
    void dims(int &t_R, int &t_C) const {
      t_R=R;
      t_C=C;
    }

    /** Returns the number of processors of the processor grid

        \return Number of processors
    */
    int size() const {
      return R*C;
    }

    /** Returns in t_R and t_C the coordinates ot the caller process in the grid AS PRESCRIBED BY THE CONCEPT
        \param[out] t_R Coordinate in first dimension
        \param[out] t_C Coordinate in second dimension
    */
    void coords(int &t_R, int &t_C) const {
      t_R = r;
      t_C = c;
    }


    /** Returns the process ID of the process with relative coordinates (I,J) with respect to the caller process AS PRESCRIBED BY THE CONCEPT
        \tparam I Relative coordinate in the first dimension
        \tparam J Relative coordinate in the seocnd dimension
        \return The process ID of the required process
    */
    template <int I, int J>
    int proc() const {
      return proc(I,J);
    }

    /** Returns the process ID of the process with relative coordinates (I,J) with respect to the caller process AS PRESCRIBED BY THE CONCEPT
        \param[in] I Relative coordinate in the first dimension
        \param[in] J Relative coordinate in the seocnd dimension
        \return The process ID of the required process
    */
    int proc(int I, int J) const {
      int rr,cc;
      if (cyclic.value0)
        rr = (R+r+I)%R;
      else
        rr = r+I;

      if (cyclic.value1)
        cc = (C+c+J)%C;
      else
        cc = c+J;

      return proc_idx(rr, cc);
    }

    /** Returns the process ID of the process with absolute coordinates specified by the input GCL::array of coordinates
        \param[in] crds GCL::aray of coordinates of the processor of which the ID is needed

        \return The process ID of the required process
    */
    int abs_proc(GCL::array<int,ndims> const & crds) const {
      return proc(crds[0]-r, crds[1]-c);
    }

  private:
    int proc_idx(int rr, int cc) const {
      if (rr >= R || rr < 0 || cc >= C || cc < 0)
        return -1;
      return rr*C+cc;
    }

  };

#ifdef _GCL_MPI_
  /** \class MPI_2D_process_grid_t
   * Class that provides a representation of a 2D process grid given an MPI CART
   * It requires the MPI CART to be defined before the grid is created
   * \tparam CYCLIC is a template argument matching \ref boollist_concept to specify periodicities
   * \n
   * This is a process grid matching the \ref proc_grid_concept concept
   */
  template <typename CYCLIC>
  struct MPI_2D_process_grid_t {

    typedef boost::true_type has_communicator;

    typedef CYCLIC period_type;

    /** number of dimensions
     */
    static const int ndims = 2;

    MPI_Comm m_communicator; // Communicator that is associated with the MPI CART!
    period_type cyclic;
    int R,C;
    int r,c;

    /** 
        Returns communicator
     */
      MPI_Comm communicator() const {
          return m_communicator;
      }

    /** Constructor that takes an MPI CART communicator, already configured, and use it to set up the process grid.
        \param c Object containing information about periodicities as defined in \ref boollist_concept
        \param comm MPI Communicator describing the MPI 2D computing grid
     */
    MPI_2D_process_grid_t(period_type const &c, MPI_Comm comm): m_communicator(comm), cyclic(c) {create(comm);}

    /** Function to create the grid. This can be called in case the grid is default constructed. Its direct use is discouraged
        \param comm MPI Communicator describing the MPI 2D computing grid
     */
    void create(MPI_Comm comm) {
      int dims[2]={0,0}, periods[2]={1,1}, coords[2]={0,0};
      MPI_Cart_get(m_communicator, 2, dims, periods, coords);
      R=dims[0];
      C=dims[1];
      r = coords[0];
      c = coords[1];
    }

    /** Returns in t_R and t_C the lenght of the dimensions of the process grid AS PRESCRIBED BY THE CONCEPT
        \param[out] t_R Number of elements in first dimension
        \param[out] t_C Number of elements in second dimension
    */
    void dims(int &t_R, int &t_C) const {
      t_R=R;
      t_C=C;
    }

    /** Returns the number of processors of the processor grid

        \return Number of processors
    */
    int size() const {
      return R*C;
    }

    /** Returns in t_R and t_C the coordinates ot the caller process in the grid AS PRESCRIBED BY THE CONCEPT
        \param[out] t_R Coordinate in first dimension
        \param[out] t_C Coordinate in second dimension
    */
    void coords(int &t_R, int &t_C) const {
      t_R = r;
      t_C = c;
    }

    /** Returns the process ID of the process with relative coordinates (I,J) with respect to the caller process AS PRESCRIBED BY THE CONCEPT
        \tparam I Relative coordinate in the first dimension
        \tparam J Relative coordinate in the second dimension
        \return The process ID of the required process
    */
    template <int I, int J>
    int proc() const {
      return proc(I,J);
    }

    /** Returns the process ID of the process with relative coordinates (I,J) with respect to the caller process AS PRESCRIBED BY THE CONCEPT
        \param[in] I Relative coordinate in the first dimension
        \param[in] J Relative coordinate in the seocnd dimension
        \return The process ID of the required process
    */
    int proc(int I, int J) const {
      int _coords[2];

      if (cyclic.value0)
        _coords[0] = (r+I)%R;
      else {
        _coords[0] = r+I;
        if (_coords[0]<0 || _coords[0]>=R)
          return -1;
      }

      if (cyclic.value1)
        _coords[1] = (c+J)%C;
      else {
        _coords[1] = c+J;
        if (_coords[1]<0 || _coords[1]>=C)
          return -1;
      }


      int res;
      MPI_Cart_rank(m_communicator, _coords, &res);
      return res;
    }

    /** Returns the process ID of the process with absolute coordinates specified by the input GCL::array of coordinates
        \param[in] crds GCL::aray of coordinates of the processor of which the ID is needed

        \return The process ID of the required process
    */
    int abs_proc(GCL::array<int,ndims> const & crds) const {
      return proc(crds[0]-r, crds[1]-c);
    }

  };

#endif

} //namespace GCL

#endif
