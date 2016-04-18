#ifndef HALO_EXCHANGE_3D_H_
#define HALO_EXCHANGE_3D_H_

//#include <boost/mpl/assert.hpp>
//#include <assert.h>
#include <mpi.h>
#include <common/gt_assert.hpp>
#include "../GCL.hpp"
#include "translate.hpp"
#include "has_communicator.hpp"
#include "helper.hpp"

/** \file
 * Pattern for regular cyclic and acyclic halo exchange pattern in 3D
 * The communicating processes are organized in a 3D grid. Given a process, neighbors processes
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
     * \tparam PROC_GRID Processor Grid type. An object of this type will be passed to constructor.
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
    class Halo_Exchange_3D {

        typedef translate_t< 3, typename default_layout_map< 3 >::type > translate;

        class sr_buffers {
            char *m_buffers[27]; // there is ona buffer more to allow for a simple indexing
            int m_size[27];      // Sizes in bytes
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
            int &size(int I, int J, int K) { return m_size[translate()(I, J, K)]; }
            int size(int I, int J, int K) const { return m_size[translate()(I, J, K)]; }
        };

        template < int I, int J, int K >
        struct TAG {
            static const int value = (K + 1) * 9 + (I + 1) * 3 + J + 1;
        };

        struct request_t {
            MPI_Request request[27];
            MPI_Request &operator()(int i, int j, int k) { return request[translate()(i, j, k)]; }
        };

        struct request_t_mark : request_t {
            bool mark[27];
            request_t_mark() {
                for (int i = 0; i < 27; ++i)
                    mark[i] = false;
            }

            bool marked(int i, int j, int k) const { return mark[translate()(i, j, k)]; }

            void set(int i, int j, int k) { mark[translate()(i, j, k)] = true; }

            void reset(int i, int j, int k) { mark[translate()(i, j, k)] = false; }
        };

        sr_buffers m_send_buffers;
        sr_buffers m_recv_buffers;
#if defined(HOSTWORKAROUND)
        sr_buffers m_host_send_buffers;
        sr_buffers m_host_recv_buffers;
#endif
        request_t request;
        request_t_mark send_request;

        const PROC_GRID /*&*/ m_proc_grid;

        template < int I, int J, int K >
        void post_receive() {
            if (m_recv_buffers.size(I, J, K)) {
#ifndef NDEBUG
                std::cout << "@" << gridtools::PID << "@ IRECV (" << I << "," << J << "," << K << ") "
                          << " P " << m_proc_grid.template proc< I, J, K >() << " - "
                          << " T " << TAG< -I, -J, -K >::value << " - "
                          << " R " << translate()(-I, -J, -K) << " - "
                          << " Amount " << m_recv_buffers.size(I, J, K) << "\n";
#endif

#ifdef GCL_TRACE
                double begin_time = MPI_Wtime();
#endif

#ifdef HOSTWORKAROUND
                // using host workaround on gpu
                // post receive to the page-locked buffer on the host
                MPI_Irecv(static_cast< char * >(m_host_recv_buffers.buffer(I, J, K)),
                    m_recv_buffers.size(I, J, K),
                    MPI_CHAR,
                    m_proc_grid.template proc< I, J, K >(),
                    TAG< -I, -J, -K >::value,
                    get_communicator(m_proc_grid),
                    &request(-I, -J, -K));
#else
                MPI_Irecv(static_cast< char * >(m_recv_buffers.buffer(I, J, K)),
                    m_recv_buffers.size(I, J, K),
                    MPI_CHAR,
                    m_proc_grid.template proc< I, J, K >(),
                    TAG< -I, -J, -K >::value,
                    get_communicator(m_proc_grid),
                    &request(-I, -J, -K));
#endif
#ifdef GCL_TRACE
                double end_time = MPI_Wtime();
                stats_collector_3D.add_event(CommEvent(ce_receive,
                    m_proc_grid.template proc< I, J, K >(),
                    TAG< -I, -J, -K >::value,
                    m_recv_buffers.size(I, J, K),
                    begin_time,
                    end_time,
                    pattern_tag));
#endif
            }
        }

        template < int I, int J, int K >
        void perform_isend() {
            if (m_send_buffers.size(I, J, K)) {
#ifndef NDEBUG
                std::cout << "@" << gridtools::PID << "@ ISEND (" << I << "," << J << "," << K << ") "
                          << " P " << m_proc_grid.template proc< I, J, K >() << " - "
                          << " T " << TAG< I, J, K >::value << " - "
                          << " R " << translate()(I, J, K) << " - "
                          << " Amount " << m_send_buffers.size(I, J, K) << "\n";
#endif

#ifdef GCL_TRACE
                double begin_time = MPI_Wtime();
#endif

#ifdef HOSTWORKAROUND
                // using host workaround on gpu
                // copy data from device to host
                cudaMemcpy(static_cast< void * >(m_host_send_buffers.buffer(I, J, K)),
                    static_cast< const void * >(m_send_buffers.buffer(I, J, K)),
                    m_host_send_buffers.size(I, J, K),
                    cudaMemcpyDeviceToHost);

                // perform send from host buffer
                MPI_Isend(static_cast< char * >(m_host_send_buffers.buffer(I, J, K)),
                    m_send_buffers.size(I, J, K),
                    MPI_CHAR,
                    m_proc_grid.template proc< I, J, K >(),
                    TAG< I, J, K >::value,
                    get_communicator(m_proc_grid),
                    &send_request(I, J, K));
#else
                MPI_Isend(static_cast< char * >(m_send_buffers.buffer(I, J, K)),
                    m_send_buffers.size(I, J, K),
                    MPI_CHAR,
                    m_proc_grid.template proc< I, J, K >(),
                    TAG< I, J, K >::value,
                    get_communicator(m_proc_grid),
                    &send_request(I, J, K));
#endif
                send_request.set(I, J, K);
#ifdef GCL_TRACE
                double end_time = MPI_Wtime();
                stats_collector_3D.add_event(CommEvent(ce_send,
                    m_proc_grid.template proc< I, J, K >(),
                    TAG< I, J, K >::value,
                    m_send_buffers.size(I, J, K),
                    begin_time,
                    end_time,
                    pattern_tag));
#endif
            }
        }

        void wait_for_sends() {
            MPI_Status status;
            for (int i = -1; i <= 1; ++i)
                for (int j = -1; j <= 1; ++j)
                    for (int k = -1; k <= 1; ++k)
                        if (send_request.marked(i, j, k)) {
#ifdef GCL_TRACE
                            double begin_time = MPI_Wtime();
#endif
                            MPI_Wait(&send_request(i, j, k), &status);
#ifdef GCL_TRACE
                            double end_time = MPI_Wtime();
                            stats_collector_3D.add_event(CommEvent(ce_send_wait,
                                // m_proc_grid.template proc<I,J,K>(),
                                m_proc_grid.proc(i, j, k),
                                // TAG<I,J,K>::value,
                                -1,
                                m_send_buffers.size(i, j, k),
                                begin_time,
                                end_time,
                                pattern_tag));
#endif
                            send_request.reset(i, j, k);
                        }
        }

        template < int I, int J, int K >
        void wait() {
            if (m_recv_buffers.size(I, J, K)) {
#ifndef NDEBUG
                std::cout << "@" << gridtools::PID << "@ WAIT  (" << I << "," << J << "," << K << ") "
                          << " R " << translate()(-I, -J, -K) << "\n";
#endif

#ifdef GCL_TRACE
                double begin_time = MPI_Wtime();
#endif

                MPI_Status status;
                MPI_Wait(&request(-I, -J, -K), &status);
#ifdef HOSTWORKAROUND
                // copy from host buffers to device
                // only need to do this if receiving from another PID
                cudaMemcpy(static_cast< void * >(m_recv_buffers.buffer(I, J, K)),
                    static_cast< const void * >(m_host_recv_buffers.buffer(I, J, K)),
                    m_host_recv_buffers.size(I, J, K),
                    cudaMemcpyHostToDevice);
#endif
#ifdef GCL_TRACE
                double end_time = MPI_Wtime();
                stats_collector_3D.add_event(CommEvent(ce_receive_wait,
                    m_proc_grid.template proc< I, J, K >(),
                    TAG< -I, -J, -K >::value,
                    m_recv_buffers.size(I, J, K),
                    begin_time,
                    end_time,
                    pattern_tag));
#endif
            }
        }

#ifdef GCL_TRACE
        int pattern_tag;
#endif

      public:
#ifdef GCL_TRACE
        void set_pattern_tag(int tag) { pattern_tag = tag; }
#endif

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
        explicit Halo_Exchange_3D(PROC_GRID /*const&*/ _pg)
            : m_send_buffers(), m_recv_buffers(), request(), send_request(), m_proc_grid(_pg)
#ifdef GCL_TRACE
              ,
              pattern_tag(-1)
#endif
        {
        }

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
        void register_send_to_buffer(void *p, int s, int I, int J, int K) {
            assert((I >= -1 && I <= 1));
            assert((J >= -1 && J <= 1));
            assert((K >= -1 && K <= 1));

#ifndef NDEBUG
//       std::cout << "@" << gridtools::PID
//                 << "@ " << __PRETTY_FUNCTION__
//                 << " : " << p << " size " << s
//                 << " I:" << I
//                 << " J:" << J
//                 << " K:" << K
//                 << " (" << translate()(I,J,K) << ")\n";
#endif

            m_send_buffers.buffer(I, J, K) = reinterpret_cast< char * >(p);
            m_send_buffers.size(I, J, K) = s;
#ifdef HOSTWORKAROUND
            // allocate a buffer on the host with page-locked memory
            m_host_send_buffers.buffer(I, J, K) = _impl::helper_alloc< char, _impl::host_page_locked >::alloc(s);
            m_host_send_buffers.size(I, J, K) = s;
#endif
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
        void register_send_to_buffer(void *p, int s) {
            BOOST_MPL_ASSERT_RELATION(I, >=, -1);
            BOOST_MPL_ASSERT_RELATION(I, <=, 1);
            BOOST_MPL_ASSERT_RELATION(J, >=, -1);
            BOOST_MPL_ASSERT_RELATION(J, <=, 1);
            BOOST_MPL_ASSERT_RELATION(K, >=, -1);
            BOOST_MPL_ASSERT_RELATION(K, <=, 1);

            register_send_to_buffer(p, s, I, J, K);
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
        void register_receive_from_buffer(void *p, int s, int I, int J, int K) {
            assert((I >= -1 && I <= 1));
            assert((J >= -1 && J <= 1));
            assert((K >= -1 && K <= 1));

#ifndef NDEBUG
//       std::cout << "@" << gridtools::PID
//                 << "@ " << __PRETTY_FUNCTION__
//                 << " : " << p << " size " << s
//                 << " I:" << I
//                 << " J:" << J
//                 << " K:" << K
//                 <<  " (" << translate()(I,J,K) << ")\n";
#endif

            m_recv_buffers.buffer(I, J, K) = reinterpret_cast< char * >(p);
            m_recv_buffers.size(I, J, K) = s;
#ifdef HOSTWORKAROUND
            // allocate a buffer on the host with page-locked memory
            m_host_recv_buffers.buffer(I, J, K) = _impl::helper_alloc< char, _impl::host_page_locked >::alloc(s);
            m_host_recv_buffers.size(I, J, K) = s;
#endif
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
        void register_receive_from_buffer(void *p, int s) {
            BOOST_MPL_ASSERT_RELATION(I, >=, -1);
            BOOST_MPL_ASSERT_RELATION(I, <=, 1);
            BOOST_MPL_ASSERT_RELATION(J, >=, -1);
            BOOST_MPL_ASSERT_RELATION(J, <=, 1);
            BOOST_MPL_ASSERT_RELATION(K, >=, -1);
            BOOST_MPL_ASSERT_RELATION(K, <=, 1);

            register_receive_from_buffer(p, s, I, J, K);
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
#ifdef HOSTWORKAROUND
            // throw an assertion because the page-locked buffer allocated in the workaround
            // has fixed size (if this is a problem we can free, then reallocate memory)
            assert(false);
#endif
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
#ifdef HOSTWORKAROUND
            // throw an assertion because the page-locked buffer allocated in the workaround
            // has fixed size (if this is a problem we can free, then reallocate memory)
            assert(false);
#endif
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
#ifdef HOSTWORKAROUND
            // throw an assertion because the page-locked buffer allocated in the workaround
            // has fixed size (if this is a problem we can free, then reallocate memory)
            assert(false);
#endif
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
            that is, send all the send-buffers to the correspondinf
            receive-buffers. When the function returns the data in receive
            buffers can be safely accessed.
         */
        void exchange() {
            start_exchange();
            wait();
        }

        void post_receives() {
            /* Posting receives face -1
             */
            if (m_proc_grid.template proc< 1, 0, -1 >() != -1) {
                post_receive< 1, 0, -1 >();
            }

            if (m_proc_grid.template proc< -1, 0, -1 >() != -1) {
                post_receive< -1, 0, -1 >();
            }

            if (m_proc_grid.template proc< 0, 1, -1 >() != -1) {
                post_receive< 0, 1, -1 >();
            }

            if (m_proc_grid.template proc< 0, -1, -1 >() != -1) {
                post_receive< 0, -1, -1 >();
            }

            /* Posting receives FOR CORNERS face -1
             */
            if (m_proc_grid.template proc< 1, 1, -1 >() != -1) {
                post_receive< 1, 1, -1 >();
            }

            if (m_proc_grid.template proc< -1, -1, -1 >() != -1) {
                post_receive< -1, -1, -1 >();
            }

            if (m_proc_grid.template proc< 1, -1, -1 >() != -1) {
                post_receive< 1, -1, -1 >();
            }

            if (m_proc_grid.template proc< -1, 1, -1 >() != -1) {
                post_receive< -1, 1, -1 >();
            }

            if (m_proc_grid.template proc< 0, 0, -1 >() != -1) {
                post_receive< 0, 0, -1 >();
            }

            /* Posting receives face 0
             */
            if (m_proc_grid.template proc< 1, 0, 0 >() != -1) {
                post_receive< 1, 0, 0 >();
            }

            if (m_proc_grid.template proc< -1, 0, 0 >() != -1) {
                post_receive< -1, 0, 0 >();
            }

            if (m_proc_grid.template proc< 0, 1, 0 >() != -1) {
                post_receive< 0, 1, 0 >();
            }

            if (m_proc_grid.template proc< 0, -1, 0 >() != -1) {
                post_receive< 0, -1, 0 >();
            }

            /* Posting receives FOR CORNERS face 0
             */
            if (m_proc_grid.template proc< 1, 1, 0 >() != -1) {
                post_receive< 1, 1, 0 >();
            }

            if (m_proc_grid.template proc< -1, -1, 0 >() != -1) {
                post_receive< -1, -1, 0 >();
            }

            if (m_proc_grid.template proc< 1, -1, 0 >() != -1) {
                post_receive< 1, -1, 0 >();
            }

            if (m_proc_grid.template proc< -1, 1, 0 >() != -1) {
                post_receive< -1, 1, 0 >();
            }

            /* Posting receives face 1
             */
            if (m_proc_grid.template proc< 1, 0, 1 >() != -1) {
                post_receive< 1, 0, 1 >();
            }

            if (m_proc_grid.template proc< -1, 0, 1 >() != -1) {
                post_receive< -1, 0, 1 >();
            }

            if (m_proc_grid.template proc< 0, 1, 1 >() != -1) {
                post_receive< 0, 1, 1 >();
            }

            if (m_proc_grid.template proc< 0, -1, 1 >() != -1) {
                post_receive< 0, -1, 1 >();
            }

            /* Posting receives FOR CORNERS face 1
             */
            if (m_proc_grid.template proc< 1, 1, 1 >() != -1) {
                post_receive< 1, 1, 1 >();
            }

            if (m_proc_grid.template proc< -1, -1, 1 >() != -1) {
                post_receive< -1, -1, 1 >();
            }

            if (m_proc_grid.template proc< 1, -1, 1 >() != -1) {
                post_receive< 1, -1, 1 >();
            }

            if (m_proc_grid.template proc< -1, 1, 1 >() != -1) {
                post_receive< -1, 1, 1 >();
            }

            if (m_proc_grid.template proc< 0, 0, 1 >() != -1) {
                post_receive< 0, 0, 1 >();
            }
        }

        void do_sends() {
            /* Sending data face -1
             */
            if (m_proc_grid.template proc< -1, 0, -1 >() != -1) {
                perform_isend< -1, 0, -1 >();
            }

            if (m_proc_grid.template proc< 1, 0, -1 >() != -1) {
                perform_isend< 1, 0, -1 >();
            }

            if (m_proc_grid.template proc< 0, -1, -1 >() != -1) {
                perform_isend< 0, -1, -1 >();
            }

            if (m_proc_grid.template proc< 0, 1, -1 >() != -1) {
                perform_isend< 0, 1, -1 >();
            }

            /* Sending data CORNERS
             */
            if (m_proc_grid.template proc< -1, -1, -1 >() != -1) {
                perform_isend< -1, -1, -1 >();
            }

            if (m_proc_grid.template proc< 1, 1, -1 >() != -1) {
                perform_isend< 1, 1, -1 >();
            }

            if (m_proc_grid.template proc< 1, -1, -1 >() != -1) {
                perform_isend< 1, -1, -1 >();
            }

            if (m_proc_grid.template proc< -1, 1, -1 >() != -1) {
                perform_isend< -1, 1, -1 >();
            }

            if (m_proc_grid.template proc< 0, 0, -1 >() != -1) {
                perform_isend< 0, 0, -1 >();
            }

            /* Sending data face 0
             */
            if (m_proc_grid.template proc< -1, 0, 0 >() != -1) {
                perform_isend< -1, 0, 0 >();
            }

            if (m_proc_grid.template proc< 1, 0, 0 >() != -1) {
                perform_isend< 1, 0, 0 >();
            }

            if (m_proc_grid.template proc< 0, -1, 0 >() != -1) {
                perform_isend< 0, -1, 0 >();
            }

            if (m_proc_grid.template proc< 0, 1, 0 >() != -1) {
                perform_isend< 0, 1, 0 >();
            }

            /* Sending data CORNERS
             */
            if (m_proc_grid.template proc< -1, -1, 0 >() != -1) {
                perform_isend< -1, -1, 0 >();
            }

            if (m_proc_grid.template proc< 1, 1, 0 >() != -1) {
                perform_isend< 1, 1, 0 >();
            }

            if (m_proc_grid.template proc< 1, -1, 0 >() != -1) {
                perform_isend< 1, -1, 0 >();
            }

            if (m_proc_grid.template proc< -1, 1, 0 >() != -1) {
                perform_isend< -1, 1, 0 >();
            }

            /* Sending data face 1
             */
            if (m_proc_grid.template proc< -1, 0, 1 >() != -1) {
                perform_isend< -1, 0, 1 >();
            }

            if (m_proc_grid.template proc< 1, 0, 1 >() != -1) {
                perform_isend< 1, 0, 1 >();
            }

            if (m_proc_grid.template proc< 0, -1, 1 >() != -1) {
                perform_isend< 0, -1, 1 >();
            }

            if (m_proc_grid.template proc< 0, 1, 1 >() != -1) {
                perform_isend< 0, 1, 1 >();
            }

            /* Sending data CORNERS
             */
            if (m_proc_grid.template proc< -1, -1, 1 >() != -1) {
                perform_isend< -1, -1, 1 >();
            }

            if (m_proc_grid.template proc< 1, 1, 1 >() != -1) {
                perform_isend< 1, 1, 1 >();
            }

            if (m_proc_grid.template proc< 1, -1, 1 >() != -1) {
                perform_isend< 1, -1, 1 >();
            }

            if (m_proc_grid.template proc< -1, 1, 1 >() != -1) {
                perform_isend< -1, 1, 1 >();
            }

            if (m_proc_grid.template proc< 0, 0, 1 >() != -1) {
                perform_isend< 0, 0, 1 >();
            }
        }

        /** When called this function initiate the data exchabge. When the
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

            post_receives();

            // UNCOMMENT THIS IF A DEADLOCK APPEARS BECAUSE SENDS HAS TO FOLLOW RECEIVES (TRUE IN SOME PLATFORMS)
            // MPI_Barrier(GSL_WORLD);

            do_sends();
        }

        void wait() {

            wait_for_sends();

            /* Actual receives face -1
             */
            if (m_proc_grid.template proc< 1, 0, -1 >() != -1) {
                wait< 1, 0, -1 >();
            }

            if (m_proc_grid.template proc< -1, 0, -1 >() != -1) {
                wait< -1, 0, -1 >();
            }

            if (m_proc_grid.template proc< 0, 1, -1 >() != -1) {
                wait< 0, 1, -1 >();
            }

            if (m_proc_grid.template proc< 0, -1, -1 >() != -1) {
                wait< 0, -1, -1 >();
            }

            if (m_proc_grid.template proc< 1, 1, -1 >() != -1) {
                wait< 1, 1, -1 >();
            }

            if (m_proc_grid.template proc< -1, -1, -1 >() != -1) {
                wait< -1, -1, -1 >();
            }

            if (m_proc_grid.template proc< -1, 1, -1 >() != -1) {
                wait< -1, 1, -1 >();
            }

            if (m_proc_grid.template proc< 1, -1, -1 >() != -1) {
                wait< 1, -1, -1 >();
            }

            if (m_proc_grid.template proc< 0, 0, -1 >() != -1) {
                wait< 0, 0, -1 >();
            }

            /* Actual receives face 0
             */
            if (m_proc_grid.template proc< 1, 0, 0 >() != -1) {
                wait< 1, 0, 0 >();
            }

            if (m_proc_grid.template proc< -1, 0, 0 >() != -1) {
                wait< -1, 0, 0 >();
            }

            if (m_proc_grid.template proc< 0, 1, 0 >() != -1) {
                wait< 0, 1, 0 >();
            }

            if (m_proc_grid.template proc< 0, -1, 0 >() != -1) {
                wait< 0, -1, 0 >();
            }

            if (m_proc_grid.template proc< 1, 1, 0 >() != -1) {
                wait< 1, 1, 0 >();
            }

            if (m_proc_grid.template proc< -1, -1, 0 >() != -1) {
                wait< -1, -1, 0 >();
            }

            if (m_proc_grid.template proc< -1, 1, 0 >() != -1) {
                wait< -1, 1, 0 >();
            }

            if (m_proc_grid.template proc< 1, -1, 0 >() != -1) {
                wait< 1, -1, 0 >();
            }

            /* Actual receives face -1
             */
            if (m_proc_grid.template proc< 1, 0, 1 >() != -1) {
                wait< 1, 0, 1 >();
            }

            if (m_proc_grid.template proc< -1, 0, 1 >() != -1) {
                wait< -1, 0, 1 >();
            }

            if (m_proc_grid.template proc< 0, 1, 1 >() != -1) {
                wait< 0, 1, 1 >();
            }

            if (m_proc_grid.template proc< 0, -1, 1 >() != -1) {
                wait< 0, -1, 1 >();
            }

            if (m_proc_grid.template proc< 1, 1, 1 >() != -1) {
                wait< 1, 1, 1 >();
            }

            if (m_proc_grid.template proc< -1, -1, 1 >() != -1) {
                wait< -1, -1, 1 >();
            }

            if (m_proc_grid.template proc< -1, 1, 1 >() != -1) {
                wait< -1, 1, 1 >();
            }

            if (m_proc_grid.template proc< 1, -1, 1 >() != -1) {
                wait< 1, -1, 1 >();
            }

            if (m_proc_grid.template proc< 0, 0, 1 >() != -1) {
                wait< 0, 0, 1 >();
            }

            // MPI_Barrier(gridtools::GCL_WORLD);
        }
    };

} // namespace gridtools

#endif
