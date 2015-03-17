#ifndef _PROC_GRIDS_3D_H_
#define _PROC_GRIDS_3D_H_

#include <string>
#include <boost/algorithm/string.hpp>
#include <iostream>
#include <cmath>
#include "../GCL.h"
#include "../../common/array.h"
#include <boost/type_traits/integral_constant.hpp>

// This file needs to be changed

namespace gridtools {

  /** \class _3D_process_grid_t
   * Class that provides a generic 3D process grid from a linear
   * process distribution.  Given a contiguos range of P processes,
   * from 0 to P-1, this class provide a distribution of these
   * processes as a 3D grid

   1) by row in the best possible aspect ratio if GCL_3D_PROC_GRID environment variable is set to BLOCKED (default).

   2) by linear layout by I, J or K if GCL_3D_PROC_GRID environment variable is set BY_I, BY_J, or BY_K, resp.

   3) others to come

   * \n
   * \tparam CYCLIC is a boollist class \link boollist_concept \endlink template parameter that specifies which dimensions of the grid are clyclic
   * \n
   * This is a process grid matching the \ref proc_grid_concept concept
   */
  template <typename CYCLIC>
  struct _3D_process_grid_t {

    /** number of dimensions
     */
    static const int ndims = 3;

    typedef boost::false_type has_communicator;
    typedef CYCLIC period_type;
    period_type cyclic;
    int R, C,S;
    int r,c,s;

  private:
    int proc_idx(int rr, int cc, int ss) const {
      if (rr >= R || rr < 0
          || cc >= C || cc < 0
          || ss >= S || ss < 0)
        return -1;
      return rr*S*C+cc*S+ss;
    }

    void compute_id_coords(int pid) {
      int my_pid = pid;
      r = my_pid/(C*S);
      int t = pid - r* C*S;
      c = t/S;
      s = t-c*S;
    }

    void create_blocked(int P, int pid) {
      double obj, oldobj;
      //oldobj = sqrt((P-r3)*(P-r3)+(1-r3)*(1-r3)+(1-r3)*(1-r3));
      oldobj = 2*P+2+2*P; // Surface

      int sq = static_cast<int>(sqrt(static_cast<double>(P)));
      for (int i=1; i<=sq; ++i) {
        int t = P/i;
        if (i*t == P) {
          int sq1 = static_cast<int>(sqrt(static_cast<double>(t)));
          for (int j=1; j<=sq1; ++j) {
            int s = t/j;
            if (j*s == t) {
              // now i, j, s is a factorization of P
              //obj=sqrt((i-r3)*(i-r3)+(j-r3)*(j-r3)+(s-r3)*(s-r3));
              obj=2*i*j+2*i*s+2*j*s; //SURFACE
              if (obj <= oldobj) {
                oldobj=obj;
                R = s;
                S = j;
                C = i;
              }
            }
          }
        }
      }

      compute_id_coords(pid);

    }

    void create_by_k(int P, int pid) {
      int sq = static_cast<int>(sqrt(static_cast<double>(P)));
      for (int i=1; i<=sq; ++i) {
        int t = P/i;
        if (i*t == P) {
          C = i;
          R = t;
        }
      }

      S = 1;

      compute_id_coords(pid);
    }

    void create_by_i(int P, int pid) {
      int sq = static_cast<int>(sqrt(static_cast<double>(P)));
      for (int i=1; i<=sq; ++i) {
        int t = P/i;
        if (i*t == P) {
          C = i;
          S = t;
        }
      }

      R = 1;

      compute_id_coords(pid);
    }

    void create_by_j(int P, int pid) {
      int sq = static_cast<int>(sqrt(static_cast<double>(P)));
      for (int i=1; i<=sq; ++i) {
        int t = P/i;
        if (i*t == P) {
          S = i;
          R = t;
        }
      }

      C = 1;

      compute_id_coords(pid);
    }

  private:
      _3D_process_grid_t( _3D_process_grid_t const& oter){
          assert(false);
      }

      _3D_process_grid_t& operator =(_3D_process_grid_t const& other){
          assert(false);
      }

  public:
    /** Constructor that takes the number of processes and the caller ID to produce the grid
        \param[in] c The object of the class used to specify periodicity in each dimension
        \param[in] P Number of processes that will make the grid
        \param[in] pid Number of processes that will make the grid
     */
    _3D_process_grid_t(period_type const &c, int P, int pid)
      : cyclic(c)
      , R(0)
      , C(0)
      , S(0)
      , r(0)
      , c(0)
      , s(0)
    {
      create(P,pid);
    }

    /** Function to create the grid. This can be called in case the
        grid is default constructed. Its direct use is discouraged.

        \param[in] P Number of processes that will make the grid
        \param[in] pid Number of processes that will make the grid
     */
    void create(int P, int pid) {
      char * pkind = getenv("GCL_3D_PROC_GRID");
      if (pkind == NULL) {
#ifndef NDEBUG
        std::cout << "Blocked grid of processes - env variable not set" << std::endl;
#endif
        create_blocked(P, pid);
        return;
      }
      std::string kind(pkind);
      boost::to_upper(kind);

      if ( kind == "BLOCKED" ) {
#ifndef NDEBUG
        std::cout << "Blocked grid of processes" << std::endl;
#endif
        create_blocked(P, pid);
      } else {
        if ( kind == "BY_I" ) {
#ifndef NDEBUG
          std::cout << "Grid of processes along directionn i" << std::endl;
#endif
          create_by_i(P, pid);
        } else {
          if ( kind == "BY_J" ) {
#ifndef NDEBUG
            std::cout << "Grid of processes along directionn j" << std::endl;
#endif
            create_by_i(P, pid);
          } else {
            if ( kind == "BY_K" ) {
#ifndef NDEBUG
              std::cout << "Grid of processes along directionn k" << std::endl;
#endif
              create_by_i(P, pid);
            } else {
              // default
#ifndef NDEBUG
              std::cout << "Blocked grid of processes (not recognized env. variable)" << std::endl;
#endif
              create_blocked(P, pid);
            }
          }
        }
      }
    }

    /** Returns in t_R and t_C the lenght of the dimensions of the process grid AS PRESCRIBED BY THE CONCEPT
        \param[out] t_R Number of elements in first dimension
        \param[out] t_C Number of elements in second dimension
        \param[out] t_S Number of elements in third dimension
    */
    void dims(int &t_R, int &t_C, int &t_S) const {
      t_R=R;
      t_C=C;
      t_S=S;
    }


    /** Returns the number of processors of the processor grid

        \return Number of processors
    */
    int size() const {
      return R*C*S;
    }


    /** Returns in t_R and t_C the coordinates ot the caller process in the grid AS PRESCRIBED BY THE CONCEPT
        \param[out] t_R Coordinate in first dimension
        \param[out] t_C Coordinate in second dimension
        \param[out] t_S Coordinate in third dimension
    */
    void coords(int &t_R, int &t_C, int &t_S) const {
      t_R = r;
      t_C = c;
      t_S = s;
    }

    /** Returns the process ID of the process with relative coordinates (I,J) with respect to the caller process AS PRESCRIBED BY THE CONCEPT
        \tparam I Relative coordinate in the first dimension
        \tparam J Relative coordinate in the seocnd dimension
        \tparam K Relative coordinate in the third dimension
        \return The process ID of the required process
    */
    template <int I, int J, int K>
    int proc() const {
      return proc(I,J,K);
    }

    /** Returns the process ID of the process with relative coordinates (I,J) with respect to the caller process AS PRESCRIBED BY THE CONCEPT
        \param[in] I Relative coordinate in the first dimension
        \param[in] J Relative coordinate in the seocnd dimension
        \param[in] K Relative coordinate in the third dimension
        \return The process ID of the required process
    */
    int proc(int I, int J, int K) const {
      int rr,cc,ss;
      if (cyclic.value(0))
        rr = (R+r+I)%R;
      else
        rr = r+I;

      if (cyclic.value(1))
        cc = (C+c+J)%C;
      else
        cc = c+J;

      if (cyclic.value(2))
        ss = (S+s+K)%S;
      else
        ss = s+K;

      return proc_idx(rr, cc, ss);
    }

    /** Returns the process ID of the process with absolute coordinates specified by the input gridtools::array of coordinates
        \param[in] crds gridtools::aray of coordinates of the processor of which the ID is needed

        \return The process ID of the required process
    */
    int abs_proc(gridtools::array<int,ndims> const & crds) const {
      return proc(crds[0]-r, crds[1]-c, crds[2]-s);
    }

  };


#ifdef _GCL_MPI_
  /** \class MPI_3D_process_grid_t
   * Class that provides a representation of a 3D process grid given an MPI CART
   * It requires the MPI CART to be defined before the grid is created
   * \tparam CYCLIC is a template argument matching \ref boollist_concept to specify periodicities
   * \n
   * This is a process grid matching the \ref proc_grid_concept concept
   */
    template <int Ndims, int SubDims=Ndims>
  struct MPI_3D_process_grid_t {

        /** number of dimensions
     */
      static const int ndims = Ndims;

    typedef boost::true_type has_communicator;
    typedef gridtools::boollist<ndims> period_type;


  private:
      MPI_Comm m_communicator; // Communicator that is associated with the MPI CART!
        period_type cyclic;
        int m_nprocs;
        int m_dimensions[ndims];
        int m_coordinates[ndims];
//       int m_boundary;
  public:
    /**
        Returns communicator
     */
      MPI_Comm communicator() const {
          return m_communicator;
      }

//   private:
      MPI_3D_process_grid_t( MPI_3D_process_grid_t const& other): m_communicator(other.m_communicator), cyclic(other.cyclic)// , m_boundary(other.m_boundary)
                                                                , m_nprocs(other.m_nprocs) {

          for (ushort_t i=0; i<ndims; ++i){
              m_dimensions[i]=other.m_dimensions[i];
              m_coordinates[i]=other.m_coordinates[i];
          }
//           assert(false);
      }

//       MPI_3D_process_grid_t& operator =(MPI_3D_process_grid_t const& other){
//           assert(false);
//       }
//   public:

    /** Constructor that takes an MPI CART communicator, already configured, and use it to set up the process grid.
        \param c Object containing information about periodicities as defined in \ref boollist_concept
        \param comm MPI Communicator describing the MPI 3D computing grid
     */
      MPI_3D_process_grid_t(period_type const &c, MPI_Comm const& comm)
          :
          m_communicator(comm),
          cyclic(c),
          m_nprocs(0)
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
              MPI_Comm_size(comm, &m_nprocs);
              if(SubDims>0)
              {
                  assert(SubDims<ndims);
                  MPI_Dims_create(m_nprocs, SubDims, m_dimensions);
                  for (ushort i=SubDims; i<ndims; ++i)
                      m_dimensions[i]=1;
              }
              else
                  MPI_Dims_create(m_nprocs, ndims, m_dimensions);
              create(comm);
          }

    /** Function to create the grid. This can be called in case the
        grid is default constructed. Its direct use is discouraged

        \param comm MPI Communicator describing the MPI 3D computing grid
     */
    void create(MPI_Comm const& comm) {
        //int dims[ndims]={0,0,0}, periods[ndims]={true,true,true}, coords[ndims]={0,0,0};
        int period[ndims];
        for (ushort_t i=0; i<ndims; ++i)
            period[i]=cyclic.value(i);
        MPI_Cart_create(comm, ndims, m_dimensions, period, false, &m_communicator);
        MPI_Cart_get(m_communicator, ndims, m_dimensions, period/*does not really care*/, m_coordinates);
//         m_boundary=0;
//         for (ushort_t i=0; i<ndims; ++i)
//             if(m_coordinates[i]==m_dimensions[i]-1) m_boundary  += std::pow(2, i);
//         for (ushort_t i=ndims; i<2*ndims; ++i)
//             if(m_coordinates[i]==0) m_boundary  += std::pow(2, i);

        // if(m_coordinates[0]==m_dimensions[0]-1) m_boundary  =  1; else m_boundary = 0;
        // if(m_coordinates[1]==m_dimensions[1]-1) m_boundary +=  2;
        // if(m_coordinates[0]==0)  m_boundary += 4;
        // if(m_coordinates[1]==0)  m_boundary += 8;

// #ifndef NDEBUG
//         printf("comunicator coordinates: [%d, %d, %d]\n",m_coordinates[0], m_coordinates[1], m_coordinates[2] );
//         printf("boundary: [%d]\n",m_boundary );
//         printf("dimensions: [%d, %d, %d]\n",m_dimensions[0], m_dimensions[1], m_dimensions[2] );
// #endif
        // for(ushort_t i=0; i<ndims; ++i)
      // {
      //     m_dimensions[i]=dims[i];
      //     m_coordinates[i] = coords[i];
      // }
    }

    /** Returns in t_R and t_C the lenght of the dimensions of the process grid AS PRESCRIBED BY THE CONCEPT
        \param[out] t_R Number of elements in first dimension
        \param[out] t_C Number of elements in second dimension
        \param[out] t_S Number of elements in third dimension
    */
      void dims(int &t_R, int &t_C, int &t_S) const {
          GRIDTOOLS_STATIC_ASSERT(ndims==3, "this interface supposes ndims=3")
              t_R=m_dimensions[0];
          t_C=m_dimensions[1];
          t_S=m_dimensions[2];
      }

      void dims(int &t_R, int &t_C) const {
          GRIDTOOLS_STATIC_ASSERT(ndims==2, "this interface supposes ndims=2")
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
        \param[out] t_S Coordinate in third dimension
    */
    void coords(int &t_R, int &t_C, int &t_S) const {
        GRIDTOOLS_STATIC_ASSERT(ndims==3, "this interface supposes ndims=3")
            t_R = m_coordinates[0];
        t_C = m_coordinates[1];
        t_S = m_coordinates[2];
    }

      void coords(int &t_R, int &t_C) const {
          GRIDTOOLS_STATIC_ASSERT(ndims==2, "this interface supposes ndims=2")
              t_R = m_coordinates[0];
          t_C = m_coordinates[1];
    }

    /** Returns the process ID of the process with relative coordinates (I,J) with respect to the caller process AS PRESCRIBED BY THE CONCEPT
        \tparam I Relative coordinate in the first dimension
        \tparam J Relative coordinate in the second dimension
        \tparam K Relative coordinate in the third dimension
        \return The process ID of the required process
    */
    template <int I, int J, int K>
        int proc() const {
        //int coords[3]={I,J,K};
        return proc(I, J, K);
    }

//       int proc(int const& I, int const& J, int const& K) const {
//           int coords[3]={I,J,K};
//           return proc(coords);
//       }

//     template <int I, int J>
//         int proc() const {
//         int coords[3]={I,J,-1};
//         return proc(coords);
//     }

      int pid() const {
          int rank;
          MPI_Comm_rank(m_communicator, &rank);
          return rank;
      }

//       int const& boundary() const {
//           return m_boundary;
//       };


    /** Returns the process ID of the process with relative coordinates (I,J) with respect to the caller process AS PRESCRIBED BY THE CONCEPT
        \param[in] I Relative coordinate in the first dimension
        \param[in] J Relative coordinate in the second dimension
        \param[in] K Relative coordinate in the third dimension
        \return The process ID of the required process
    */
      int proc(int I, int J, int K) const {
      int _coords[3];

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

      if (cyclic.value(2))
        _coords[2] = (m_coordinates[2]+K)%m_dimensions[2];
      else {
        _coords[2] = m_coordinates[2]+K;
        if (_coords[2]<0 || _coords[2]>=m_dimensions[2])
          return -1;
      }

      int pid=0;
      MPI_Comm_rank(MPI_COMM_WORLD, &pid);
      int res;
      MPI_Cart_rank(m_communicator, _coords, &res);
#ifndef NDEBUG
      std::cout<<"communication from: "<<pid<<"along["<<I<<", "<<J<<", "<<K<<"]" <<std::endl;
      std::cout<<" cyclic: "<<cyclic.value(0)<<" "<<cyclic.value(1)<<" "<<cyclic.value(2)<<"]"<<std::endl;
      std::cout<<"m_coordinates:[ "<<m_coordinates[0]<<" "<<m_coordinates[1]<<" "<<m_coordinates[2]<<"]"<<std::endl;
      std::cout<<"m_dimensions:[ "<<m_dimensions[0]<<" "<<m_dimensions[1]<<" "<<m_dimensions[2]<<"]"<<std::endl;
      std::cout<<"_coords:[ "<<_coords[0]<<" "<<_coords[1]<<" "<<_coords[2]<<"]"<<std::endl;
      std::cout<<"result: "<<res<<std::endl;
#endif
      return res;
    }

    /** Returns the process ID of the process with relative coordinates (I,J) with respect to the caller process AS PRESCRIBED BY THE CONCEPT
        \param[in] I Relative coordinate in the first dimension
        \param[in] J Relative coordinate in the second dimension
        \param[in] K Relative coordinate in the third dimension
        \return The process ID of the required process
    */
//       int proc(int const* coords) const {
//           int _coords[ndims];
//           int res;

//           for (ushort_t i=0; i<ndims; ++i)
//               if (cyclic.value(i)) {
//                   _coords[i] = (m_coordinates[i]+coords[i])%m_dimensions[i];
//               }
//               else {
//                   _coords[i] = m_coordinates[i]+coords[i];
//                   if(_coords[i]>=0 && _coords[i]<m_dimensions[i])
//                       return -1;
//               }

//           printf("getting rank. [I,J,K] :[%d, %d, %d]\n\n\n", coords[0],coords[1],coords[2]);
//           printf("getting rank. m_dimensions:[%d, %d, %d]\n", m_dimensions[0], m_dimensions[1], m_dimensions[2]);
//           printf("getting rank. m_coordinates:[%d, %d, %d]\n", m_coordinates[0], m_coordinates[1], m_coordinates[2]);
//           printf("getting rank. _coords:[%d, %d, %d]\n\n\n", _coords[0], _coords[1], _coords[2]);
//           MPI_Cart_rank(m_communicator, _coords, &res);
//           return res;
//       }

      int const* coordinates()const {return m_coordinates;}
      int const* dimensions()const {return m_dimensions;}

    /** Returns the process ID of the process with absolute coordinates specified by the input gridtools::array of coordinates
        \param[in] crds gridtools::aray of coordinates of the processor of which the ID is needed

        \return The process ID of the required process
    */
      int abs_proc(gridtools::array<int,ndims> const & crds) const {
          return proc(crds[0]-m_coordinates[0], crds[1]-m_coordinates[1], crds[2]-m_coordinates[2]);
    }

      int ntasks(){return m_nprocs;}

      bool periodic (int index) const {
          assert(index<ndims);
          return cyclic.value(index);
      }

      int const& coordinates(ushort_t const& i)const {return m_coordinates[i];}
      int const& dimensions(ushort_t const& i)const {return m_dimensions[i];}

  };

#endif

} //namespace gridtools

#endif
