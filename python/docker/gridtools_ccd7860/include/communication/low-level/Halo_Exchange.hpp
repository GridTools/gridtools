#ifndef _HALO_EXCHANGE_H_
#define _HALO_EXCHANGE_H_

// #include <boost/preprocessor/repetition/enum.hpp>
// #include <boost/preprocessor/arithmetic/inc.hpp>
// #include <boost/preprocessor/repetition/enum_params.hpp>
// #include <boost/preprocessor/repetition/repeat.hpp>
// #include <boost/fusion/include/vector.hpp>
// #include <boost/fusion/include/fold.hpp>

#define IMINUS_TAG       0
#define IPLUS_TAG        1
#define JMINUS_TAG       2
#define JPLUS_TAG        3
#define KMINUS_TAG       4
#define KPLUS_TAG        5
#define IMINUSJPLUSKPLUS_TAG  10
#define IPLUSJPLUSKPLUS_TAG   11
#define IMINUSJMINUSKPLUS_TAG 12
#define IPLUSJMINUSKPLUS_TAG  13
#define IMINUSJPLUSKMINUS_TAG  14
#define IPLUSJPLUSKMINUS_TAG   15
#define IMINUSJMINUSKMINUS_TAG 16
#define IPLUSJMINUSKMINUS_TAG  17
#define IMINUSJPLUS_TAG  18
#define IPLUSJPLUS_TAG   19
#define IMINUSJMINUS_TAG 20
#define IPLUSJMINUS_TAG  21
#define IMINUSKPLUS_TAG  22
#define IPLUSKPLUS_TAG   23
#define IMINUSKMINUS_TAG 24
#define IPLUSKMINUS_TAG  25
#define JMINUSKPLUS_TAG  26
#define JPLUSKPLUS_TAG   27
#define JMINUSKMINUS_TAG 28
#define JPLUSKMINUS_TAG  29

// #define PRE_EDGE 400
// #define PRE_GHOST 2
// #define SIZE_OF_TYPE 8

namespace gridtools {
  /* here we store the buckets data structure, first target is 2D
   */
  template <typename PROC_GRID, int ALIGN_SIZE=1> // ALIGN_SIZE is the size of the types used. Need to specify better what it is. It is needed to allow send of receive to other types (like MPI types) to use to send data.
  // there migh be a mpi_type<8>::value to be MPI_DOUBLE and mpi_type<8>::divisor as the value to divide the legnth of the buffer in bytes to compute sizes correctly. this is just a proposal.
  struct Halo_Exchange_2D {};

  template <typename PROC_GRID, int ALIGN_SIZE=1>
  struct Halo_Exchange_3D {};

}

#endif
