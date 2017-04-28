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
#ifndef _PROC_GRIDS_3D_H_
#define _PROC_GRIDS_3D_H_

#include <string>
#include <boost/algorithm/string.hpp>
#include <iostream>
#include <cmath>
#include "../GCL.hpp"
#include "../../common/array.hpp"
#include "../../common/boollist.hpp"
#include <boost/type_traits/integral_constant.hpp>

// This file needs to be changed

namespace gridtools {

#ifdef _GCL_MPI_
    /** \class MPI_3D_process_grid_t
     * Class that provides a representation of a 3D process grid given an MPI CART
     * It requires the MPI CART to be defined before the grid is created
     * \tparam CYCLIC is a template argument matching \ref boollist_concept to specify periodicities
     * \n
     * This is a process grid matching the \ref proc_grid_concept concept
     */
    template < int Ndims >
    struct MPI_3D_process_grid_t {

        /** number of dimensions
         */
        static const int ndims = Ndims;

        typedef boost::true_type has_communicator;
        typedef gridtools::boollist< ndims > period_type;

      private:
        MPI_Comm m_communicator; // Communicator that is associated with the MPI CART!
        period_type m_cyclic;
        int m_nprocs;
        gridtools::array< int, ndims > m_dimensions;
        gridtools::array< int, ndims > m_coordinates;

      public:
        MPI_3D_process_grid_t(MPI_3D_process_grid_t const &other)
            : m_communicator(other.m_communicator), m_cyclic(other.cyclic()), m_nprocs(other.m_nprocs) {
            for (ushort_t i = 0; i < ndims; ++i) {
                m_dimensions[i] = other.m_dimensions[i];
                m_coordinates[i] = other.m_coordinates[i];
            }
        }

        /** Constructor that takes an MPI CART communicator, already configured, and use it to set up the process grid.
            \param c Object containing information about periodicities as defined in \ref boollist_concept
            \param comm MPI Communicator describing the MPI 3D computing grid
        */
        MPI_3D_process_grid_t(period_type const &c, MPI_Comm const &comm)
            : m_communicator(comm), m_cyclic(c), m_nprocs(0), m_dimensions(), m_coordinates() {

            MPI_Cart_get(comm, ndims, &m_dimensions[0], 0x0, &m_coordinates[0]);
        }

        /**
           Returns communicator
        */
        MPI_Comm communicator() const { return m_communicator; }

        /**@brief wrapper around MPI_Dims_Create checking the array size*/
        static int dims_create(int const &procs_, int const &ndims_, array< int, ndims > &dims_array_) {
            assert(ndims >= ndims_);
            return MPI_Dims_create(procs_, ndims_, &dims_array_[0]);
        }

        /** Function to create the grid. This can be called in case the
            grid is default constructed. Its direct use is discouraged

            \param comm MPI Communicator describing the MPI 3D computing grid
        */
        void create(MPI_Comm const &comm) {
            // int dims[ndims]={0,0,0}, periods[ndims]={true,true,true}, coords[ndims]={0,0,0};
            int period[ndims];
            for (ushort_t i = 0; i < ndims; ++i)
                period[i] = m_cyclic.value(i);
            MPI_Cart_create(comm, ndims, &m_dimensions[0], period, false, &m_communicator);
            MPI_Cart_get(m_communicator, ndims, &m_dimensions[0], period /*does not really care*/, &m_coordinates[0]);
        }

        /** Returns in t_R and t_C the lenght of the dimensions of the process grid AS PRESCRIBED BY THE CONCEPT
            \param[out] t_R Number of elements in first dimension
            \param[out] t_C Number of elements in second dimension
            \param[out] t_S Number of elements in third dimension
        */
        void dims(int &t_R, int &t_C, int &t_S) const {
            GRIDTOOLS_STATIC_ASSERT(ndims == 3, "this interface supposes ndims=3");
            t_R = m_dimensions[0];
            t_C = m_dimensions[1];
            t_S = m_dimensions[2];
        }

        void dims(int &t_R, int &t_C) const {
            GRIDTOOLS_STATIC_ASSERT(ndims == 2, "this interface supposes ndims=2");
            t_R = m_dimensions[0];
            t_C = m_dimensions[1];
        }

        /** Returns the number of processors of the processor grid

            \return Number of processors
        */
        uint_t size() const {
            uint_t ret = m_dimensions[0];
            for (ushort_t i = 1; i < ndims; ++i)
                ret *= m_dimensions[i];
            return ret;
        }

        /** Returns in t_R and t_C the coordinates ot the caller process in the grid AS PRESCRIBED BY THE CONCEPT
            \param[out] t_R Coordinate in first dimension
            \param[out] t_C Coordinate in second dimension
            \param[out] t_S Coordinate in third dimension
        */
        void coords(int &t_R, int &t_C, int &t_S) const {
            GRIDTOOLS_STATIC_ASSERT(ndims == 3, "this interface supposes ndims=3");
            t_R = m_coordinates[0];
            t_C = m_coordinates[1];
            t_S = m_coordinates[2];
        }

        void coords(int &t_R, int &t_C) const {
            GRIDTOOLS_STATIC_ASSERT(ndims == 2, "this interface supposes ndims=2");
            t_R = m_coordinates[0];
            t_C = m_coordinates[1];
        }

        /** Returns the process ID of the process with relative coordinates (I,J) with respect to the caller process AS
           PRESCRIBED BY THE CONCEPT
            \tparam I Relative coordinate in the first dimension
            \tparam J Relative coordinate in the second dimension
            \tparam K Relative coordinate in the third dimension
            \return The process ID of the required process
        */
        template < int I, int J, int K >
        int proc() const {
            // int coords[3]={I,J,K};
            return proc(I, J, K);
        }

        int pid() const {
            int rank;
            MPI_Comm_rank(m_communicator, &rank);
            return rank;
        }

        /** Returns the process ID of the process with relative coordinates (I,J) with respect to the caller process AS
           PRESCRIBED BY THE CONCEPT
            \param[in] I Relative coordinate in the first dimension
            \param[in] J Relative coordinate in the second dimension
            \param[in] K Relative coordinate in the third dimension
            \return The process ID of the required process
        */
        int proc(int I, int J, int K) const {
            int _coords[3];

            if (m_cyclic.value(0))
                _coords[0] = (m_coordinates[0] + I) % m_dimensions[0];
            else {
                _coords[0] = m_coordinates[0] + I;
                if (_coords[0] < 0 || _coords[0] >= m_dimensions[0])
                    return -1;
            }

            if (m_cyclic.value(1))
                _coords[1] = (m_coordinates[1] + J) % m_dimensions[1];
            else {
                _coords[1] = m_coordinates[1] + J;
                if (_coords[1] < 0 || _coords[1] >= m_dimensions[1])
                    return -1;
            }

            if (m_cyclic.value(2))
                _coords[2] = (m_coordinates[2] + K) % m_dimensions[2];
            else {
                _coords[2] = m_coordinates[2] + K;
                if (_coords[2] < 0 || _coords[2] >= m_dimensions[2])
                    return -1;
            }

            int pid = 0;
            MPI_Comm_rank(MPI_COMM_WORLD, &pid);
            int res;
            MPI_Cart_rank(m_communicator, _coords, &res);
            return res;
        }

        GT_FUNCTION
        gridtools::array< int, ndims > const &coordinates() const { return m_coordinates; }

        GT_FUNCTION
        gridtools::array< int, ndims > const &dimensions() const { return m_dimensions; }

        /** Returns the process ID of the process with absolute coordinates specified by the input gridtools::array of
           coordinates
            \param[in] crds gridtools::aray of coordinates of the processor of which the ID is needed

            \return The process ID of the required process
        */
        int abs_proc(gridtools::array< int, ndims > const &crds) const {
            return proc(crds[0] - m_coordinates[0], crds[1] - m_coordinates[1], crds[2] - m_coordinates[2]);
        }

        int ntasks() { return m_nprocs; }

        bool periodic(int index) const {
            assert(index < ndims);
            return m_cyclic.value(index);
        }

        array< bool, ndims > periodic() const {
            GRIDTOOLS_STATIC_ASSERT(period_type::m_size == ndims, "Dimensions not matching");
            return m_cyclic.value();
        }

        period_type const &cyclic() const { return m_cyclic; }

        int const &coordinates(ushort_t const &i) const { return m_coordinates[i]; }
        int const &dimensions(ushort_t const &i) const { return m_dimensions[i]; }
    };

#endif

} // namespace gridtools

#endif
