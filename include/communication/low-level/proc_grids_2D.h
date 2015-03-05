#ifndef _PROC_GRIDS_2D_H_
#define _PROC_GRIDS_2D_H_

#include <iostream>
#include <cmath>
#include "../GCL.h"
#include "../../common/array.h"
#include <boost/type_traits/integral_constant.hpp>

// This file needs to be changed

/** \namespace gridtools
 * All library classes, functions, and objects will reside in this namespace.
 */
namespace gridtools {

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
      if (cyclic.value(0))
        rr = (R+r+I)%R;
      else
        rr = r+I;

      if (cyclic.value(1))
        cc = (C+c+J)%C;
      else
        cc = c+J;

      return proc_idx(rr, cc);
    }

    /** Returns the process ID of the process with absolute coordinates specified by the input gridtools::array of coordinates
        \param[in] crds gridtools::aray of coordinates of the processor of which the ID is needed

        \return The process ID of the required process
    */
    int abs_proc(gridtools::array<int,ndims> const & crds) const {
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
      static const int ndims = CYCLIC::size;

      MPI_Comm m_communicator; // Communicator that is associated with the MPI CART!
      period_type cyclic;
      int m_dimensions[ndims];
      int m_coordinates[ndims];
      int m_nprocs;

    /**
        Returns communicator
     */
      MPI_Comm communicator() const {
          return m_communicator;
      }

//   private:
      MPI_2D_process_grid_t( MPI_2D_process_grid_t const& other): m_communicator(other.m_communicator), cyclic(other.cyclic)// , m_boundary(other.m_boundary)
                                                                , m_nprocs(other.m_nprocs) {
              for (ushort_t i=0; i<ndims; ++i){
                  m_dimensions[i]=0;
                  m_coordinates[i]=0;
              }
              create(m_communicator);
//           assert(false);
//, m_dimensions(other.m_dimensions), m_coordinates(other.m_coordinates)
      }

    /** Constructor that takes an MPI CART communicator, already configured, and use it to set up the process grid.
        \param c Object containing information about periodicities as defined in \ref boollist_concept
        \param comm MPI Communicator describing the MPI 2D computing grid
     */
      MPI_2D_process_grid_t(period_type const &c, MPI_Comm const& comm)
          :
          cyclic(c)
#if  !defined(__clang__) && defined(CXX11_ENABLED)
          ,m_dimensions{0},
          m_coordinates{0}
#endif
          {
#if defined(__clang__) || !defined(CXX11_ENABLED)
              for (ushort_t i=0; i<ndims; ++i){
                  m_dimensions[i]=0;
                  m_coordinates[i]=0;
              }
#endif
              create(comm);}

    /** Function to create the grid. This can be called in case the grid is default constructed. Its direct use is discouraged
        \param comm MPI Communicator describing the MPI 2D computing grid
     */
    void create(MPI_Comm comm) {
        int period[ndims];
        for (ushort_t i=0; i<ndims; ++i)
            period[i]=cyclic.value(i);
        MPI_Comm_size(comm, &m_nprocs);
        MPI_Dims_create(m_nprocs, ndims, m_dimensions);
        MPI_Cart_create(comm, ndims, m_dimensions, period, false, &m_communicator);
        MPI_Cart_get(m_communicator, ndims, m_dimensions, period/*does not really care*/, m_coordinates);
    }

    /** Returns in t_R and t_C the lenght of the dimensions of the process grid AS PRESCRIBED BY THE CONCEPT
        \param[out] t_R Number of elements in first dimension
        \param[out] t_C Number of elements in second dimension
    */
    void dims(int &t_R, int &t_C) const {
        GRIDTOOLS_STATIC_ASSERT(ndims==2, "this interface supposes ndims=3")
            t_R=m_dimensions[0];
        t_C=m_dimensions[1];
    }

    /** Returns the number of processors of the processor grid

        \return Number of processors
    */
      uint_t size() const {
        uint_t ret=m_dimensions[0];
        for (ushort_t i=1; i< ndims; ++i)
            ret *= m_dimensions[i];
        return ret;
    }

    /** Returns in t_R and t_C the coordinates ot the caller process in the grid AS PRESCRIBED BY THE CONCEPT
        \param[out] t_R Coordinate in first dimension
        \param[out] t_C Coordinate in second dimension
    */

      void coords(int &t_R, int &t_C) const {
          GRIDTOOLS_STATIC_ASSERT(ndims==2, "this interface supposes ndims=2")
              t_R = m_coordinates[0];
          t_C = m_coordinates[1];
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

      if (cyclic.value(0))
        _coords[0] = (m_coordinates[0]+I)%m_dimensions[0];
      else {
        _coords[0] = m_coordinates[0]+I;
        if (_coords[0]<0 || _coords[0]>=m_dimensions[0])
          return -1;
      }

      if (cyclic.value(1))
        _coords[1] = (m_coordinates[1]+J)%m_dimensions[1];
      else {
        _coords[1] = m_coordinates[1]+J;
        if (_coords[1]<0 || _coords[1]>=m_dimensions[1])
          return -1;
      }


      int res;
      MPI_Cart_rank(m_communicator, _coords, &res);
      return res;
    }

    /** Returns the process ID of the process with absolute coordinates specified by the input gridtools::array of coordinates
        \param[in] crds gridtools::aray of coordinates of the processor of which the ID is needed

        \return The process ID of the required process
    */
    int abs_proc(gridtools::array<int,ndims> const & crds) const {
      return proc(crds[0]-m_coordinates[0], crds[1]-m_coordinates[1]);
    }

      int pid() const {
          int rank;
          MPI_Cart_rank(m_communicator, m_coordinates, &rank);
          return rank;
      }

      int const* coordinates()const {return m_coordinates;}
      int const* dimensions()const {return m_dimensions;}
      int const& coordinates(ushort_t const& i)const {return m_coordinates[i];}
      int const& dimensions(ushort_t const& i)const {return m_dimensions[i];}

  };

#endif

} //namespace gridtools

#endif
