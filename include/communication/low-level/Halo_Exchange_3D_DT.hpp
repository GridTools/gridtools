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
#ifndef HALO_EXCHANGE_3D_DT_H_
#define HALO_EXCHANGE_3D_DT_H_

#include <common/gt_assert.hpp>
#include "../GCL.hpp"
#include "translate.hpp"
#include "has_communicator.hpp"

/** \file
 * Pattern for regular cyclic and acyclic halo exchange pattern in 3D
 * The communicating processes are arganized in a 3D grid. Given a process, neighbors processes
 * are located using relative coordinates. In the next diagram, the given process is (0,0,0)
 * while the neighbors are indicated with their relative coordinates.
 * \code
 *       ----------------------------------
 *       |          |          |          |
 *       | -1,-1,-1 | -1,0,-1  | -1,1,-1  |
 *       |          |          |          |
 *       ----------------------------------
 *       |          |          |          |
 *       |  0,-1,-1 |  0,0,-1  |  0,1,-1  |
 *       |          |          |          |
 *       ----------------------------------
 *       |          |          |          |
 *       |  1,-1,-1 |  1,0,-1  |  1,1,-1  |
 *       |          |          |          |
 *       ----------------------------------
 *
 *       ----------------------------------
 *       |          |          |          |
 *       | -1,-1, 0 | -1,0, 0  | -1,1, 0  |
 *       |          |          |          |
 *       ----------------------------------
 *       |          |          |          |
 *       |  0,-1, 0 |  0,0, 0  |  0,1, 0  |
 *       |          |          |          |
 *       ----------------------------------
 *       |          |          |          |
 *       |  1,-1, 0 |  1,0, 0  |  1,1, 0  |
 *       |          |          |          |
 *       ----------------------------------
 *
 *       ----------------------------------
 *       |          |          |          |
 *       | -1,-1, 1 | -1,0, 1  | -1,1, 1  |
 *       |          |          |          |
 *       ----------------------------------
 *       |          |          |          |
 *       |  0,-1, 1 |  0,0, 1  |  0,1, 1  |
 *       |          |          |          |
 *       ----------------------------------
 *       |          |          |          |
 *       |  1,-1, 1 |  1,0, 1  |  1,1, 1  |
 *       |          |          |          |
 *       ----------------------------------
 * \endcode
 */

namespace gridtools {

    /** \class Halo_Exchange_3D
     * Class to instantiate, define and run a regular cyclic and acyclic
     * halo exchange pattern in 3D.  By regular it is intended that the
     * amount of data sent and received during the execution of the
     * pattern is known by all participants to the comunciation without
     * communication. More specifically, the ampunt of data received is
     * decided before the execution of the pattern. If a different
     * ampunt of data is received from some process the behavior is
     * undefined.\n
     * Given a process (i,j,k), we can define \f$s_{ijk}^{mnl}\f$ and
     * \f$r_{ijk}^{mnl}\f$ as the data sent and received from process
     * (i,j,k) to/from process (i+m, j+n, k+l), respectively. For this pattern
     * m, n and l are supposed to be in the range -1, 0, +1. \n\n When
     * executing the Halo_Exchange_3D pattern, the requirement is that
     * \f[r_{ijk}^{mnl} = s_{i+m,j+n,k+l}^{-m,-n,-l}\f].
     * \n
     * \tparam PROG_GRID Processor Grid type. An object of this type will be passed to constructor.
     * \tparam ALIGN integer parameter that specify the alignment of the data to used. UNUSED IN CURRENT VERSION
     * \n\n\n
     * Pattern for regular cyclic and acyclic halo exchange pattern in 3D
     * The communicating processes are arganized in a 3D grid. Given a process, neighbors processes
     * are located using relative coordinates. In the next diagram, the given process is (0,0,0)
     * while the neighbors are indicated with their relative coordinates.
     * \code
     *       ----------------------------------
     *       |          |          |          |
     *       | -1,-1,-1 | -1,0,-1  | -1,1,-1  |
     *       |          |          |          |
     *       ----------------------------------
     *       |          |          |          |
     *       |  0,-1,-1 |  0,0,-1  |  0,1,-1  |
     *       |          |          |          |
     *       ----------------------------------
     *       |          |          |          |
     *       |  1,-1,-1 |  1,0,-1  |  1,1,-1  |
     *       |          |          |          |
     *       ----------------------------------
     *
     *       ----------------------------------
     *       |          |          |          |
     *       | -1,-1, 0 | -1,0, 0  | -1,1, 0  |
     *       |          |          |          |
     *       ----------------------------------
     *       |          |          |          |
     *       |  0,-1, 0 |  0,0, 0  |  0,1, 0  |
     *       |          |          |          |
     *       ----------------------------------
     *       |          |          |          |
     *       |  1,-1, 0 |  1,0, 0  |  1,1, 0  |
     *       |          |          |          |
     *       ----------------------------------
     *
     *       ----------------------------------
     *       |          |          |          |
     *       | -1,-1, 1 | -1,0, 1  | -1,1, 1  |
     *       |          |          |          |
     *       ----------------------------------
     *       |          |          |          |
     *       |  0,-1, 1 |  0,0, 1  |  0,1, 1  |
     *       |          |          |          |
     *       ----------------------------------
     *       |          |          |          |
     *       |  1,-1, 1 |  1,0, 1  |  1,1, 1  |
     *       |          |          |          |
     *       ----------------------------------
     * \endcode
     * The pattern is cyclic or not bepending on the process grid passed
     * to it. The cyclicity may be on only one dimension.
     * An example of use of the pattern is given below
     \code
     OUT CODE HERE AS IN 2D CASE
     \endcode

     A running example can be found in the included example. \ example Halo_Exchange_test_3D.cpp
    */
    template < typename PROC_GRID, int ALIGN = 1 >
    class Halo_Exchange_3D_DT {

        typedef translate_t< 3, typename default_layout_map< 3 >::type > translate;

        class sr_buffers {
            char *m_buffers[27];         // there is ona buffer more to allow for a simple indexing
            MPI_Datatype m_datatype[27]; // there is ona buffer more to allow for a simple indexing
            int m_size[27];              // Sizes in bytes
          public:
            explicit sr_buffers() {
                m_buffers[0] = NULL;
                m_buffers[1] = NULL;
                m_buffers[2] = NULL;
                m_buffers[3] = NULL;
                m_buffers[4] = NULL;
                m_buffers[5] = NULL;
                m_buffers[6] = NULL;
                m_buffers[7] = NULL;
                m_buffers[8] = NULL;
                m_buffers[9] = NULL;
                m_buffers[10] = NULL;
                m_buffers[11] = NULL;
                m_buffers[12] = NULL;
                m_buffers[13] = NULL;
                m_buffers[14] = NULL;
                m_buffers[15] = NULL;
                m_buffers[16] = NULL;
                m_buffers[17] = NULL;
                m_buffers[18] = NULL;
                m_buffers[19] = NULL;
                m_buffers[20] = NULL;
                m_buffers[21] = NULL;
                m_buffers[22] = NULL;
                m_buffers[23] = NULL;
                m_buffers[24] = NULL;
                m_buffers[25] = NULL;
                m_buffers[26] = NULL;

                m_size[0] = 0;
                m_size[1] = 0;
                m_size[2] = 0;
                m_size[3] = 0;
                m_size[4] = 0;
                m_size[5] = 0;
                m_size[6] = 0;
                m_size[7] = 0;
                m_size[8] = 0;
                m_size[9] = 0;
                m_size[10] = 0;
                m_size[11] = 0;
                m_size[12] = 0;
                m_size[13] = 0;
                m_size[14] = 0;
                m_size[15] = 0;
                m_size[16] = 0;
                m_size[17] = 0;
                m_size[18] = 0;
                m_size[19] = 0;
                m_size[20] = 0;
                m_size[21] = 0;
                m_size[22] = 0;
                m_size[23] = 0;
                m_size[24] = 0;
                m_size[25] = 0;
                m_size[26] = 0;
            }

            char *&buffer(int I, int J, int K) { return m_buffers[translate()(I, J, K)]; }
            MPI_Datatype &datatype(int I, int J, int K) { return m_datatype[translate()(I, J, K)]; }
            int &size(int I, int J, int K) { return m_size[translate()(I, J, K)]; }
            int size(int I, int J, int K) const { return m_size[translate()(I, J, K)]; }
        };

        struct TAG {
            static const int value(int I, int J, int K) { return (K + 1) * 9 + (I + 1) * 3 + J + 1; }
        };

        struct request_t {
            MPI_Request request[27];
            MPI_Request &operator()(int i, int j, int k) { return request[translate()(i, j, k)]; }
        };

        const PROC_GRID m_proc_grid;

        sr_buffers m_send_buffers;
        sr_buffers m_recv_buffers;
        request_t request;
        request_t send_request;

        template < int I, int J, int K >
        void post_receive() {
            return post_receive(I, J, K);
        }

        void post_receive(int I, int J, int K) {
            if (m_recv_buffers.size(I, J, K)) {
#ifndef NDEBUG
                int ss2;
                MPI_Pack_size(1, m_recv_buffers.datatype(I, J, K), gridtools::GCL_WORLD, &ss2);
                std::cout << "@" << gridtools::PID << "@ IRECV (" << I << "," << J << "," << K << ") "
                          << " P " << m_proc_grid.proc(I, J, K) << " - "
                          << " T " << TAG::value(-I, -J, -K) << " - "
                          << " R " << translate()(-I, -J, -K) << " - "
                          << " Amount " << ss2 << "\n";
#endif
                MPI_Irecv(static_cast< char * >(m_recv_buffers.buffer(I, J, K)),
                    m_recv_buffers.size(I, J, K),
                    m_recv_buffers.datatype(I, J, K),
                    m_proc_grid.proc(I, J, K),
                    TAG::value(-I, -J, -K),
                    get_communicator(m_proc_grid),
                    &request(-I, -J, -K));
            }
        }

        template < int I, int J, int K >
        void perform_isend() {
            return perform_isend(I, J, K);
        }

        void perform_isend(int I, int J, int K) {
            if (m_send_buffers.size(I, J, K)) {
#ifndef NDEBUG
                int ss2;
                MPI_Pack_size(1, m_send_buffers.datatype(I, J, K), gridtools::GCL_WORLD, &ss2);
                std::cout << "@" << gridtools::PID << "@ ISEND (" << I << "," << J << "," << K << ") "
                          << " P " << m_proc_grid.proc(I, J, K) << " - "
                          << " T " << TAG::value(I, J, K) << " - "
                          << " R " << translate()(I, J, K) << " - "
                          << " Amount " << ss2 << "\n";
#endif
                MPI_Isend(static_cast< char * >(m_send_buffers.buffer(I, J, K)),
                    m_send_buffers.size(I, J, K),
                    m_send_buffers.datatype(I, J, K),
                    m_proc_grid.proc(I, J, K),
                    TAG::value(I, J, K),
                    get_communicator(m_proc_grid),
                    &send_request(I, J, K));
            }
        }

        template < int I, int J, int K >
        void wait() {
            if (m_recv_buffers.size(I, J, K)) {
#ifndef NDEBUG
                std::cout << "@" << gridtools::PID << "@ WAIT  (" << I << "," << J << "," << K << ") "
                          << " R " << translate()(-I, -J, -K) << "\n";
#endif

                MPI_Status status;
                MPI_Wait(&request(-I, -J, -K), &status);
            }
        }

      public:
        /** Type of the processor grid used by the pattern
         */
        typedef PROC_GRID grid_type;

        /** Type of the translation map to map processors to buffers.
         */
        typedef translate translate_type;

        /** Constructor that takes the process grid. Must be executed by all the processes in the grid.
         * It is not possible to change the process grid once the pattern has beeninstantiated.
         *
         */
        explicit Halo_Exchange_3D_DT(PROC_GRID _pg)
            : m_proc_grid(_pg), m_send_buffers(), m_recv_buffers(), request(), send_request() {}

        /** Function to retrieve the grid from the pattern, from which user can query
            location information.

            If used to get process grid information additional information can be
            found in \link GRIDS_INTERACTION \endlink
        */
        PROC_GRID const &proc_grid() const { return m_proc_grid; }

        /** Function to register send buffers with the communication patter.

          Values I and J are coordinates relative to calling process and
          the buffer is the container for the data to be sent to that
          process. The amount of data is specified as number of bytes. It
          is possible to override the previous pointer by re-registering a
          new pointer with a given destination.

           \param[in] p Pointer to the first element of type T to send
           \param[in] s Number of bytes (not number of elements) to be send. In any case this is the amount of data
          sent.
           \param[in] I Relative coordinates of the receiving process along the first dimension
           \param[in] J Relative coordinates of the receiving process along the second dimension
           \param[in] K Relative coordinates of the receiving process along the third dimension
        */
        void register_send_to_buffer(void *p, MPI_Datatype const &DT, int s, int I, int J, int K) {
            assert((I >= -1 && I <= 1));
            assert((J >= -1 && J <= 1));
            assert((K >= -1 && K <= 1));

            // #ifndef NDEBUG
            //       std::cout << "@" << gridtools::PID
            //                 << "@ " << __PRETTY_FUNCTION__
            //                 << " : " << p << " size " << s
            //                 << " I:" << I
            //                 << " J:" << J
            //                 << " K:" << K
            //                 << " (" << translate()(I,J,K) << ")\n";
            // #endif

            m_send_buffers.buffer(I, J, K) = reinterpret_cast< char * >(p);
            m_send_buffers.datatype(I, J, K) = DT;
            m_send_buffers.size(I, J, K) = s;
        }

        /** Function to register send buffers with the communication patter.

           Values I, J and K are coordinates relative to calling process
           and the buffer is the container for the data to be sent to that
           process. The amount of data is specified as number of bytes. It
           is possible to override the previous pointer by re-registering
           a new pointer with a given destination.

           \tparam I Relative coordinates of the receiving process along the first dimension
           \tparam J Relative coordinates of the receiving process along the second dimension
           \tparam K Relative coordinates of the receiving process along the third dimension
           \param[in] p Pointer to the first element of type T to send
           \param[in] s Number of bytes (not number of elements) to be send. In any case this is the amount of data
           sent.
        */
        template < int I, int J, int K >
        void register_send_to_buffer(void *p, MPI_Datatype const &DT, int s) {
            BOOST_MPL_ASSERT_RELATION(I, >=, -1);
            BOOST_MPL_ASSERT_RELATION(I, <=, 1);
            BOOST_MPL_ASSERT_RELATION(J, >=, -1);
            BOOST_MPL_ASSERT_RELATION(J, <=, 1);
            BOOST_MPL_ASSERT_RELATION(K, >=, -1);
            BOOST_MPL_ASSERT_RELATION(K, <=, 1);

            register_send_to_buffer(p, DT, s, I, J, K);
        }

        /** Function to register buffers for received data with the communication patter.

           Values I, J and K are coordinates relative to calling process and
           the buffer is the container for the data to be received from
           that process. The amount of data is specified as number of
           bytes. It is possible to override the previous pointer by
           re-registering a new pointer with a given source.

           \param[in] p Pointer to the first element of type T  where to put received data

           \param[in] s Number of bytes (not number of elements) expected
           to be received. This is the data that is assumed to arrive. If
           less data arrives, the behaviour is undefined.
           \param[in] I Relative coordinates of the receiving process along the first dimension
           \param[in] J Relative coordinates of the receiving process along the second dimension
           \param[in] K Relative coordinates of the receiving process along the third dimension
        */
        void register_receive_from_buffer(void *p, MPI_Datatype const &DT, int s, int I, int J, int K) {
            assert((I >= -1 && I <= 1));
            assert((J >= -1 && J <= 1));
            assert((K >= -1 && K <= 1));

            // #ifndef NDEBUG
            //       std::cout << "@" << gridtools::PID
            //                 << "@ " << __PRETTY_FUNCTION__
            //                 << " : " << p << " size " << s
            //                 << " I:" << I
            //                 << " J:" << J
            //                 << " K:" << K
            //                 <<  " (" << translate()(I,J,K) << ")\n";
            // #endif

            m_recv_buffers.buffer(I, J, K) = reinterpret_cast< char * >(p);
            m_recv_buffers.datatype(I, J, K) = DT;
            m_recv_buffers.size(I, J, K) = s;
        }

        /** Function to register buffers for received data with the communication patter.

           Values I, J and K are coordinates relative to calling process and
           the buffer is the container for the data to be received from
           that process. The amount of data is specified as number of
           bytes. It is possible to override the previous pointer by
           re-registering a new pointer with a given source.

           \tparam I Relative coordinates of the receiving process along the first dimension
           \tparam J Relative coordinates of the receiving process along the second dimension
           \tparam K Relative coordinates of the receiving process along the third dimension
           \param[in] p Pointer to the first element of type T where to put received data
           \param[in] s Number of bytes (not number of elements) expected
           to be received. This is the data that is assumed to arrive. If
           less data arrives, the behaviour is undefined.
        */
        template < int I, int J, int K >
        void register_receive_from_buffer(void *p, MPI_Datatype const &DT, int s) {
            BOOST_MPL_ASSERT_RELATION(I, >=, -1);
            BOOST_MPL_ASSERT_RELATION(I, <=, 1);
            BOOST_MPL_ASSERT_RELATION(J, >=, -1);
            BOOST_MPL_ASSERT_RELATION(J, <=, 1);
            BOOST_MPL_ASSERT_RELATION(K, >=, -1);
            BOOST_MPL_ASSERT_RELATION(K, <=, 1);

            register_receive_from_buffer(p, DT, s, I, J, K);
        }

        /* Setting sizes */

        /** Function to set send buffers sizes if the size must be updated
            from a previous registration. The same pointer passed during
            registration will be used to send data. It is possible to
            override the previous pointer by re-registering a new pointer
            with a given destination.

           Values I, J and K are coordinates relative to calling process and
           the buffer is the container for the data to be sent to that
           process. The amount of data is specified as number of bytes.

           \param[in] s Number of bytes (not number of elements) to be sent.
           \param[in] I Relative coordinates of the receiving process along the first dimension
           \param[in] J Relative coordinates of the receiving process along the second dimension
           \param[in] K Relative coordinates of the receiving process along the third dimension
        */
        void set_send_to_size(int s, int I, int J, int K) {
            assert((I >= -1 && I <= 1));
            assert((J >= -1 && J <= 1));
            assert((K >= -1 && K <= 1));

            m_send_buffers.size(I, J, K) = s;
        }

        /** Function to set send buffers sizes if the size must be updated
            from a previous registration. The same pointer passed during
            registration will be used to send data. It is possible to
            override the previous pointer by re-registering a new pointer
            with a given destination.

           Values I, J and K are coordinates relative to calling process and
           the buffer is the container for the data to be sent to that
           process. The amount of data is specified as number of bytes.

           \tparam I Relative coordinates of the receiving process along the first dimension
           \tparam J Relative coordinates of the receiving process along the second dimension
           \tparam K Relative coordinates of the receiving process along the third dimension
           \param[in] s Number of bytes (not number of elements) to be sent.
        */
        template < int I, int J, int K >
        void set_send_to_size(int s) const {
            BOOST_MPL_ASSERT_RELATION(I, >=, -1);
            BOOST_MPL_ASSERT_RELATION(I, <=, 1);
            BOOST_MPL_ASSERT_RELATION(J, >=, -1);
            BOOST_MPL_ASSERT_RELATION(J, <=, 1);
            BOOST_MPL_ASSERT_RELATION(K, >=, -1);
            BOOST_MPL_ASSERT_RELATION(K, <=, 1);

            set_send_to_size(s, I, J, K);
        }

        /** Function to set receive buffers sizes if the size must be
            updated from a previous registration. The same pointer passed
            during registration will be used to receive data. It is
            possible to override the previous pointer by re-registering a
            new pointer with a given source.

            Values I, J and K are coordinates relative to calling process and
            the buffer is the container for the data to be sent to that
            process. The amount of data is specified as number of bytes.

            \param[in] s Number of bytes (not number of elements) to be packed.
            \param[in] I Relative coordinates of the receiving process along the first dimension
            \param[in] J Relative coordinates of the receiving process along the second dimension
            \param[in] K Relative coordinates of the receiving process along the third dimension
        */
        void set_receive_from_size(int s, int I, int J, int K) const {
            assert((I >= -1 && I <= 1));
            assert((J >= -1 && J <= 1));
            assert((K >= -1 && K <= 1));

            m_send_buffers.size(I, J, K) = s;
        }

        /** Function to set receive buffers sizes if the size must be
            updated from a previous registration. The same pointer passed
            during registration will be used to receive data. It is
            possible to override the previous pointer by re-registering a
            new pointer with a given source.

            Values I and J are coordinates relative to calling process and
            the buffer is the container for the data to be sent to that
            process. The amount of data is specified as number of bytes.

            \tparam I Relative coordinates of the receiving process along the first dimension
            \tparam J Relative coordinates of the receiving process along the second dimension
            \tparam K Relative coordinates of the receiving process along the third dimension
            \param[in] s Number of bytes (not number of elements) to be packed.
        */
        template < int I, int J, int K >
        void set_receive_from_size(int s) const {
            BOOST_MPL_ASSERT_RELATION(I, >=, -1);
            BOOST_MPL_ASSERT_RELATION(I, <=, 1);
            BOOST_MPL_ASSERT_RELATION(J, >=, -1);
            BOOST_MPL_ASSERT_RELATION(J, <=, 1);
            BOOST_MPL_ASSERT_RELATION(K, >=, -1);
            BOOST_MPL_ASSERT_RELATION(K, <=, 1);

            set_receive_from_size(s, I, J, K);
        }

        /** Retrieve the size of the buffer containing data to be sent to neighbor I, J, K.

            \tparam I Relative coordinates of the receiving process along the first dimension
            \tparam J Relative coordinates of the receiving process along the second dimension
            \tparam K Relative coordinates of the receiving process along the third dimension
        */
        int send_size(int I, int J, int K) const { return m_send_buffers.size(I, J, K); }

        /** Retrieve the size of the buffer containing data to be received from neighbor I, J, K.

            \tparam I Relative coordinates of the receiving process along the first dimension
            \tparam J Relative coordinates of the receiving process along the second dimension
            \tparam K Relative coordinates of the receiving process along the third dimension
        */
        int recv_size(int I, int J, int K) const { return m_recv_buffers.size(I, J, K); }

        /** When called this function executes the communication pattern,
            that is, send all the send-buffers to the corresponding
            receive-buffers. When the function returns the data in receive
            buffers can be safely accessed.
         */
        void exchange() {
            start_exchange();
            wait();
        }

        /** When called this function initiate the data exchange. When the
            function returns the data has to be considered already to be
            transfered. Buffers should not be considered safe to access
            until the wait() function returns.
         */
        void start_exchange() {

            //      cout << GSL_pid() << " proc coords: " << r << " " << c << endl;
            /* NORTH/IMINUS
                     |---------| |---------| |---------| |---------| |---------| |---------| |---------| |---------|
                     |         | |         | |      |  | |  |      | |      |  | |  |      | |-------  | |  -------|
                     |         | |---------| |      |  | |  |      | | r<R-1|  | |  | r<R-1| |      |  | |  |      |
                     |  r<R-1  | |         | | c<C-1|  | |  |      | | c<C-1|  | |  | c>0  | | r>0  |  | |  | r>0  |
               WEST  |         | |   r>0   | |      |  | |  | c>0  | |      |  | |  |      | | c<C-1|  | |  | c>0  |
               EAST
               JMINUS|---------| |         | |      |  | |  |      | |      |  | |  |      | |      |  | |  | |JPLUS
                     |         | |         | |      |  | |  |      | |-------  | |  -------| |      |  | |  |      |
                     |---------| |---------| |---------| |---------| |---------| |---------| |---------| |---------|
               SOUTH/IPLUS
            */

            /* order of neighbors for sends and receives, all processes use the same order */
            static int ord[26][3] = {/* faces */
                {0, 0, -1},
                {-1, 0, 0},
                {1, 0, 0},
                {0, -1, 0},
                {0, 1, 0},
                {0, 0, 1},
                /* lines */
                {1, 0, -1},
                {-1, 0, -1},
                {0, 1, -1},
                {0, -1, -1},
                {1, 1, 0},
                {-1, -1, 0},
                {1, -1, 0},
                {-1, 1, 0},
                {1, 0, 1},
                {-1, 0, 1},
                {0, 1, 1},
                {0, -1, 1},
                /* corner points */
                {1, 1, -1},
                {-1, -1, -1},
                {1, -1, -1},
                {-1, 1, -1},
                {1, 1, 1},
                {-1, -1, 1},
                {1, -1, 1},
                {-1, 1, 1}};

            /* Posting receives
            */
            for (int l = 0; l < 26; l++) {
                int i = ord[l][0];
                int j = ord[l][1];
                int k = ord[l][2];
                if (m_proc_grid.proc(i, j, k) != -1) {
                    post_receive(i, j, k);
                }
            }

            // UNCOMMENT THIS IF A DEADLOCK APPEARS BECAUSE SENDS HAS TO FOLLOW RECEIVES (TRUE IN SOME PLATFORMS)
            // MPI_Barrier(GSL_WORLD);

            /* Doing sends
            */
            for (int l = 0; l < 26; l++) {
                int i = ord[l][0];
                int j = ord[l][1];
                int k = ord[l][2];
                if (m_proc_grid.proc(i, j, k) != -1) {
                    perform_isend(i, j, k);
                }
            }
        }

        void wait() {

            /* order of neighbors for doing the waits, all processes use the same order */
            static int ord[26][3] = {/* faces */
                {0, 0, -1},
                {-1, 0, 0},
                {1, 0, 0},
                {0, -1, 0},
                {0, 1, 0},
                {0, 0, 1},
                /* lines */
                {1, 0, -1},
                {-1, 0, -1},
                {0, 1, -1},
                {0, -1, -1},
                {1, 1, 0},
                {-1, -1, 0},
                {1, -1, 0},
                {-1, 1, 0},
                {1, 0, 1},
                {-1, 0, 1},
                {0, 1, 1},
                {0, -1, 1},
                /* corner points */
                {1, 1, -1},
                {-1, -1, -1},
                {1, -1, -1},
                {-1, 1, -1},
                {1, 1, 1},
                {-1, -1, 1},
                {1, -1, 1},
                {-1, 1, 1}};

            for (int l = 0; l < 26; l++) {
                int i = ord[l][0];
                int j = ord[l][1];
                int k = ord[l][2];
                if (m_proc_grid.proc(i, j, k) != -1) {
                    wait(i, j, k);
                }
            }

            // MPI_Barrier(gridtools::GCL_WORLD);
        }
    };

} // namespace gridtools

#endif
