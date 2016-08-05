/*
  GridTools Libraries

  Copyright (c) 2016, GridTools Consortium
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
#ifndef _HALO_EXCHANGE_2D_H
#define _HALO_EXCHANGE_2D_H

#include <boost/mpl/assert.h>
#include <common/gt_assert.hpp>
#include "../GCL.hpp"
#include "translate.hpp"
#include "has_communicator.hpp"

/** \file
 * Pattern for regular cyclic and acyclic halo exchange pattern in 2D
 * The communicating processes are arganized in a 2D grid. Given a process, neighbors processes
 * are located using relative coordinates. In the next diagram, the given process is (0,0)
 * while the neighbors are indicated with their relative coordinates.
 * \code
 *       -------------------------
 *       |       |       |       |
 *       | -1,-1 | -1,0  | -1,1  |
 *       |       |       |       |
 *       -------------------------
 *       |       |       |       |
 *       |  0,-1 |  0,0  |  0,1  |
 *       |       |       |       |
 *       -------------------------
 *       |       |       |       |
 *       |  1,-1 |  1,0  |  1,1  |
 *       |       |       |       |
 *       -------------------------
 * \endcode
 */

/** \namespace gridtools
 * All library classes, functions, and objects will reside in this namespace.
 */
namespace gridtools {

    /** \class Halo_Exchange_2D
     * Class to instantiate, define and run a regular cyclic and acyclic
     * halo exchange pattern in 2D. By regular it is intended that the
     * amount of data sent and received during the execution of the
     * pattern is known by all participants to the comunciation without
     * communication. More specifically, the ampunt of data received is
     * decided before the execution of the pattern. If a different
     * ampunt of data is received from some process the behavior is
     * undefined.\n
     * Given a process (i,j), we can define \f$s_{ij}^{mn}\f$ and
     * \f$r_{ij}^{mn}\f$ as the data sent and received from process
     * (i,j) to/from process (i+m, j+n), respectively. For this pattern
     * m and n are supposed to be in the range -1, 0, +1. \n\n When
     * executing the Halo_Exchange_2D pattern, the requirement is that
     * \f[r_{ij}^{mn} = s_{i+m,j+n}^{-m,-n}\f].
     * \n
     * \tparam PROG_GRID Processor Grid type. An object of this type will be passed to constructor.
     * \tparam ALIGN integer parameter that specify the alignment of the data to used. UNUSED IN CURRENT VERSION
     * \n\n\n
     * Pattern for regular cyclic and acyclic halo exchange pattern in 2D
     * The communicating processes are arganized in a 2D grid. Given a process, neighbors processes
     * are located using relative coordinates. In the next diagram, the given process is (0,0)
     * while the neighbors are indicated with their relative coordinates.
     * \code
     *       -------------------------
     *       |       |       |       |
     *       | -1,-1 | -1,0  | -1,1  |
     *       |       |       |       |
     *       -------------------------
     *       |       |       |       |
     *       |  0,-1 |  0,0  |  0,1  |
     *       |       |       |       |
     *       -------------------------
     *       |       |       |       |
     *       |  1,-1 |  1,0  |  1,1  |
     *       |       |       |       |
     *       -------------------------
     * \endcode
     * The pattern is cyclic or not bepending on the process grid passed to it. The cyclicity may be on only one
     dimension.
     * An example of use of the pattern is given below
     \code
       int iminus;
       int iplus;
       int jminus;
       int jplus;
       int iminusjminus;
       int iplusjminus;
       int iminusjplus;
       int iplusjplus;

       int iminus_r;
       int iplus_r;
       int jminus_r;
       int jplus_r;
       int iminusjminus_r;
       int iplusjminus_r;
       int iminusjplus_r;
       int iplusjplus_r;

       typedef gridtools::_2D_proc_grid_t grid_type;

       grid_type pg(P,my_id);

       gridtools::Halo_Exchange_2D<grid_type> he(pg);

       he.register_send_to_buffer<-1,-1>(&iminusjminus, sizeof(int));
       he.register_send_to_buffer<-1, 1>(&iminusjplus, sizeof(int));
       he.register_send_to_buffer< 1,-1>(&iplusjminus, sizeof(int));
       he.register_send_to_buffer< 1, 1>(&iplusjplus, sizeof(int));
       he.register_send_to_buffer<-1, 0>(&iminus, sizeof(int));
       he.register_send_to_buffer< 1, 0>(&iplus, sizeof(int));
       he.register_send_to_buffer< 0,-1>(&jminus, sizeof(int));
       he.register_send_to_buffer< 0, 1>(&jplus, sizeof(int));

       he.register_receive_from_buffer<-1,-1>(&iminusjminus_r, sizeof(int));
       he.register_receive_from_buffer<-1, 1>(&iminusjplus_r, sizeof(int));
       he.register_receive_from_buffer< 1,-1>(&iplusjminus_r, sizeof(int));
       he.register_receive_from_buffer< 1, 1>(&iplusjplus_r, sizeof(int));
       he.register_receive_from_buffer<-1, 0>(&iminus_r, sizeof(int));
       he.register_receive_from_buffer< 1, 0>(&iplus_r, sizeof(int));
       he.register_receive_from_buffer< 0,-1>(&jminus_r, sizeof(int));
       he.register_receive_from_buffer< 0, 1>(&jplus_r, sizeof(int));

       he.exchange();
       \endcode


       A running example can be found in the included example. \example test_halo_exchange_3D.cpp \example
     test_halo_exchange_2D.cpp
     */
    template < typename PROC_GRID, int ALIGN = 1 >
    class Halo_Exchange_2D {

        typedef translate_t< 2, typename default_layout_map< 2 >::type > translate;

        class sr_buffers {
            char *m_buffers[9]; // there is ona buffer more to allow for a simple indexing
            int m_size[9];      // Sizes in bytes

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
                m_size[0] = 0;
                m_size[1] = 0;
                m_size[2] = 0;
                m_size[3] = 0;
                m_size[4] = 0;
                m_size[5] = 0;
                m_size[6] = 0;
                m_size[7] = 0;
                m_size[8] = 0;
            }

            char *&buffer(int I, int J) { return m_buffers[translate()(I, J)]; }
            int &size(int I, int J) { return m_size[translate()(I, J)]; }
            int size(int I, int J) const { return m_size[translate()(I, J)]; }
        };

        template < int I, int J >
        struct TAG {
            static const int value = (I + 1) * 3 + J + 1;
        };

        struct request_t {
            MPI_Request request[9];
            MPI_Request &operator()(int i, int j) { return request[translate()(i, j)]; }
        };

        const PROC_GRID m_proc_grid;
        sr_buffers m_send_buffers;
        sr_buffers m_recv_buffers;
        request_t request;
        request_t send_request;

        template < int I, int J >
        void post_receive() {
#ifndef NDEBUG
            std::cout << "@" << gridtools::PID << "@ IRECV from (" << I << "," << J << ") "
                      << " P " << m_proc_grid.template proc< I, J >() << " - "
                      << " T " << TAG< -I, -J >::value << " - "
                      << " R " << translate()(-I, -J) << " Amount " << m_recv_buffers.size(I, J) << "\n";
#endif

            MPI_Irecv(static_cast< char * >(m_recv_buffers.buffer(I, J)),
                m_recv_buffers.size(I, J),
                MPI_CHAR,
                m_proc_grid.template proc< I, J >(),
                TAG< -I, -J >::value,
                get_communicator(m_proc_grid),
                &request(-I, -J));
        }

        template < int I, int J >
        void perform_isend() {
#ifndef NDEBUG
            std::cout << "@" << gridtools::PID << "@ ISEND to   (" << I << "," << J << ") "
                      << " P " << m_proc_grid.template proc< I, J >() << " - "
                      << " T " << TAG< I, J >::value << " - "
                      << " R " << translate()(I, J) << " Amount " << m_send_buffers.size(I, J) << "\n";
#endif
            MPI_Isend(static_cast< char * >(m_send_buffers.buffer(I, J)),
                m_send_buffers.size(I, J),
                MPI_CHAR,
                m_proc_grid.template proc< I, J >(),
                TAG< I, J >::value,
                get_communicator(m_proc_grid),
                &send_request(I, J));
        }

        template < int I, int J >
        void wait() {
#ifndef NDEBUG
            std::cout << "@" << gridtools::PID << "@ WAIT  (" << I << "," << J << ") "
                      << " R " << translate()(-I, -J) << "\n";
#endif

            MPI_Status status;
            MPI_Wait(&request(-I, -J), &status);
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
        explicit Halo_Exchange_2D(PROC_GRID _pg)
            : m_proc_grid(_pg), m_send_buffers(), m_recv_buffers(), request(), send_request() {}

        /** Returns the processor grid (as const reference) been used in construction

            If used to get process grid information additional information can be
            found in \link GRIDS_INTERACTION \endlink
         */
        PROC_GRID const &proc_grid() const { return m_proc_grid; }

        /** Function to register send buffers with the communication patter.
           Values I and J are coordinates relative to calling process and the buffer is the container for the data to be
           sent to that process. The amount of data is specified as number of bytes. It is possible to override the
           previous pointer by re-registering a new pointer with a given destination.
           \param[in] p Pointer to the first element of type T to send
           \param[in] s Number of bytes (not number of elements) to be send. In any case this is the amount of data
           sent.
           \param[in] I Relative coordinates of the receiving process along the first dimension
           \param[in] J Relative coordinates of the receiving process along the second dimension
        */
        void register_send_to_buffer(void *p, int s, int I, int J) {
            assert((I >= -1 && I <= 1));
            assert((J >= -1 && J <= 1));

#ifndef NDEBUG
            std::cout << "@" << gridtools::PID << "@ " << __PRETTY_FUNCTION__ << " : " << p << " size " << s
                      << " I:" << I << " J:" << J << " (" << translate()(I, J) << ")\n";
#endif

            m_send_buffers.buffer(I, J) = reinterpret_cast< char * >(p);
            m_send_buffers.size(I, J) = s;
        }

        /** Function to register send buffers with the communication patter.
           Values I and J are coordinates relative to calling process and the buffer is the container for the data to be
           sent to that process. The amount of data is specified as number of bytes. It is possible to override the
           previous pointer by re-registering a new pointer with a given destination.
           \tparam I Relative coordinates of the receiving process along the first dimension
           \tparam J Relative coordinates of the receiving process along the second dimension
           \param[in] p Pointer to the first element of type T to send
           \param[in] s Number of bytes (not number of elements) to be send. In any case this is the amount of data
           sent.
        */
        template < int I, int J >
        void register_send_to_buffer(void *p, int s) {
            BOOST_MPL_ASSERT_RELATION(I, >=, -1);
            BOOST_MPL_ASSERT_RELATION(I, <=, 1);
            BOOST_MPL_ASSERT_RELATION(J, >=, -1);
            BOOST_MPL_ASSERT_RELATION(J, <=, 1);

            register_send_to_buffer(p, s, I, J);
        }

        /** Function to register buffers for received data with the communication patter.
           Values I and J are coordinates relative to calling process and the buffer is the container for the data to be
           received from that process. The amount of data is specified as number of bytes. It is possible to override
           the previous pointer by re-registering a new pointer with a given source.
           \param[in] p Pointer to the first element of type T  where to put received data
           \param[in] s Number of bytes (not number of elements) expected to be received. This is the data that is
           assumed to arrive. If less data arrives, the behaviour is undefined.
           \param[in] I Relative coordinates of the receiving process along the first dimension
           \param[in] J Relative coordinates of the receiving process along the second dimension
        */
        void register_receive_from_buffer(void *p, int s, int I, int J) {
            assert((I >= -1 && I <= 1));
            assert((J >= -1 && J <= 1));

#ifndef NDEBUG
            std::cout << "@" << gridtools::PID << "@ " << __PRETTY_FUNCTION__ << " : " << p << " size " << s
                      << " I:" << I << " J:" << J << " (" << translate()(I, J) << ")\n";
#endif

            m_recv_buffers.buffer(I, J) = reinterpret_cast< char * >(p);
            m_recv_buffers.size(I, J) = s;
        }

        /** Function to register buffers for received data with the communication patter.
           Values I and J are coordinates relative to calling process and the buffer is the container for the data to be
           received from that process. The amount of data is specified as number of bytes. It is possible to override
           the previous pointer by re-registering a new pointer with a given source.
           \tparam I Relative coordinates of the receiving process along the first dimension
           \tparam J Relative coordinates of the receiving process along the second dimension
           \param[in] p Pointer to the first element of type T where to put received data
           \param[in] s Number of bytes (not number of elements) expected to be received. This is the data that is
           assumed to arrive. If less data arrives, the behaviour is undefined.
        */
        template < int I, int J >
        void register_receive_from_buffer(void *p, int s) {
            BOOST_MPL_ASSERT_RELATION(I, >=, -1);
            BOOST_MPL_ASSERT_RELATION(I, <=, 1);
            BOOST_MPL_ASSERT_RELATION(J, >=, -1);
            BOOST_MPL_ASSERT_RELATION(J, <=, 1);

            register_receive_from_buffer(p, s, I, J);
        }

        /* Setting sizes */

        /** Function to set send buffers sizes if the size must be updated from a previous registration. The same
           pointer passed during registration will be used to send data. It is possible to override the previous pointer
           by re-registering a new pointer with a given destination.
           Values I and J are coordinates relative to calling process and the buffer is the container for the data to be
           sent to that process. The amount of data is specified as number of bytes.
           \param[in] s Number of bytes (not number of elements) to be sent.
           \param[in] I Relative coordinates of the receiving process along the first dimension
           \param[in] J Relative coordinates of the receiving process along the second dimension
        */
        void set_send_to_size(int s, int I, int J) {
            assert((I >= -1 && I <= 1));
            assert((J >= -1 && J <= 1));

            m_send_buffers.size(I, J) = s;
        }

        /** Function to set send buffers sizes if the size must be updated from a previous registration. The same
           pointer passed during registration will be used to send data. It is possible to override the previous pointer
           by re-registering a new pointer with a given destination.
           Values I and J are coordinates relative to calling process and the buffer is the container for the data to be
           sent to that process. The amount of data is specified as number of bytes.
           \tparam I Relative coordinates of the receiving process along the first dimension
           \tparam J Relative coordinates of the receiving process along the second dimension
           \param[in] s Number of bytes (not number of elements) to be sent.
        */
        template < int I, int J >
        void set_send_to_size(int s) const {
            BOOST_MPL_ASSERT_RELATION(I, >=, -1);
            BOOST_MPL_ASSERT_RELATION(I, <=, 1);
            BOOST_MPL_ASSERT_RELATION(J, >=, -1);
            BOOST_MPL_ASSERT_RELATION(J, <=, 1);

            set_send_to_size(s, I, J);
        }

        /** Function to set receive buffers sizes if the size must be updated from a previous registration. The same
           pointer passed during registration will be used to receive data. It is possible to override the previous
           pointer by re-registering a new pointer with a given source.
           Values I and J are coordinates relative to calling process and the buffer is the container for the data to be
           sent to that process. The amount of data is specified as number of bytes.
           \param[in] s Number of bytes (not number of elements) to be packed.
           \param[in] I Relative coordinates of the receiving process along the first dimension
           \param[in] J Relative coordinates of the receiving process along the second dimension
        */
        void set_receive_from_size(int s, int I, int J) const {
            assert((I >= -1 && I <= 1));
            assert((J >= -1 && J <= 1));

            m_send_buffers.size(I, J) = s;
        }

        /** Function to set receive buffers sizes if the size must be updated from a previous registration. The same
           pointer passed during registration will be used to receive data. It is possible to override the previous
           pointer by re-registering a new pointer with a given source.
           Values I and J are coordinates relative to calling process and the buffer is the container for the data to be
           sent to that process. The amount of data is specified as number of bytes.
           \tparam I Relative coordinates of the receiving process along the first dimension
           \tparam J Relative coordinates of the receiving process along the second dimension
           \param[in] s Number of bytes (not number of elements) to be packed.
        */
        template < int I, int J >
        void set_receive_from_size(int s) const {
            BOOST_MPL_ASSERT_RELATION(I, >=, -1);
            BOOST_MPL_ASSERT_RELATION(I, <=, 1);
            BOOST_MPL_ASSERT_RELATION(J, >=, -1);
            BOOST_MPL_ASSERT_RELATION(J, <=, 1);

            set_receive_from_size(s, I, J);
        }

        /** Retrieve the size of the buffer containing data to be sent to neighbor I, J.

            \tparam I Relative coordinates of the receiving process along the first dimension
            \tparam J Relative coordinates of the receiving process along the second dimension
        */
        int send_size(int I, int J) const { return m_send_buffers.size(I, J); }

        /** Retrieve the size of the buffer containing data to be received from neighbor I, J.

            \tparam I Relative coordinates of the receiving process along the first dimension
            \tparam J Relative coordinates of the receiving process along the second dimension
        */
        int recv_size(int I, int J) const { return m_recv_buffers.size(I, J); }

        /** When called this function executes the communication pattern, that is, send all the send-buffers to the
         * correspondinf receive-buffers. When the function returns the data in receive buffers can be safely accessed.
         */
        void exchange() {

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

            /* Posting receives
             */
            if (m_proc_grid.template proc< 1, 0 >() != -1) {
                post_receive< 1, 0 >();
            }

            if (m_proc_grid.template proc< -1, 0 >() != -1) {
                post_receive< -1, 0 >();
            }

            if (m_proc_grid.template proc< 0, 1 >() != -1) {
                post_receive< 0, 1 >();
            }

            if (m_proc_grid.template proc< 0, -1 >() != -1) {
                post_receive< 0, -1 >();
            }

            /* Posting receives FOR CORNERS
             */
            if (m_proc_grid.template proc< 1, 1 >() != -1) {
                post_receive< 1, 1 >();
            }

            if (m_proc_grid.template proc< -1, -1 >() != -1) {
                post_receive< -1, -1 >();
            }

            if (m_proc_grid.template proc< 1, -1 >() != -1) {
                post_receive< 1, -1 >();
            }

            if (m_proc_grid.template proc< -1, 1 >() != -1) {
                post_receive< -1, 1 >();
            }

            // UNCOMMENT THIS IF A DEADLOCK APPEARS BECAUSE SENDS HAS TO FOLLOW RECEIVES (TRUE IN SOME PLATFORMS)
            // MPI_Barrier(GCL_WORLD);

            /* Sending data
             */
            if (m_proc_grid.template proc< -1, 0 >() != -1) {
                perform_isend< -1, 0 >();
            }

            if (m_proc_grid.template proc< 1, 0 >() != -1) {
                perform_isend< 1, 0 >();
            }

            if (m_proc_grid.template proc< 0, -1 >() != -1) {
                perform_isend< 0, -1 >();
            }

            if (m_proc_grid.template proc< 0, 1 >() != -1) {
                perform_isend< 0, 1 >();
            }

            /* Sending data CORNERS
             */
            if (m_proc_grid.template proc< -1, -1 >() != -1) {
                perform_isend< -1, -1 >();
            }

            if (m_proc_grid.template proc< 1, 1 >() != -1) {
                perform_isend< 1, 1 >();
            }

            if (m_proc_grid.template proc< 1, -1 >() != -1) {
                perform_isend< 1, -1 >();
            }

            if (m_proc_grid.template proc< -1, 1 >() != -1) {
                perform_isend< -1, 1 >();
            }

            /* Actual receives
             */
            if (m_proc_grid.template proc< 1, 0 >() != -1) {
                wait< 1, 0 >();
            }

            if (m_proc_grid.template proc< -1, 0 >() != -1) {
                wait< -1, 0 >();
            }

            if (m_proc_grid.template proc< 0, 1 >() != -1) {
                wait< 0, 1 >();
            }

            if (m_proc_grid.template proc< 0, -1 >() != -1) {
                wait< 0, -1 >();
            }

            if (m_proc_grid.template proc< 1, 1 >() != -1) {
                wait< 1, 1 >();
            }

            if (m_proc_grid.template proc< -1, -1 >() != -1) {
                wait< -1, -1 >();
            }

            if (m_proc_grid.template proc< -1, 1 >() != -1) {
                wait< -1, 1 >();
            }

            if (m_proc_grid.template proc< 1, -1 >() != -1) {
                wait< 1, -1 >();
            }
        }
    };

} // namespace gridtools

#endif
