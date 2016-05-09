#pragma once
#include "partitioner.hpp"

/**
@file
@brief Simple Partitioner Class
This file defines a simple partitioner splitting a structured cartesian grid

Hypotheses:
- tensor product grid structure
- halo region is the same on both sizes in one direction, but can differ in every direction
- there is no padding, i.e. the first address index "begin" coincides with the halo in that direction. Same for the last
address index.

The partitioner class is storage-agnostic, does not know anything about the storage, and thus the same partitioner can
be applied to multiple storages
*/

namespace gridtools {

    template < typename TopologyType >
    class cell_topology;

    template < typename T >
    struct is_cell_topology : boost::false_type {};

    template < typename TopologyType >
    struct is_cell_topology< cell_topology< TopologyType > > : boost::true_type {};

    // use of static polimorphism (partitioner methods may be accessed from whithin loops)
    template < typename GridTopology, typename Communicator >
    class partitioner_trivial : public partitioner< partitioner_trivial< GridTopology, Communicator > > {
      public:
        typedef GridTopology topology_t;
        typedef Communicator communicator_t;
        typedef partitioner< partitioner_trivial< GridTopology, Communicator > > super;

        using super::LOW;
        using super::UP;

        static const ushort_t space_dimensions = topology_t::space_dimensions;

        GRIDTOOLS_STATIC_ASSERT(is_cell_topology< GridTopology >::value,
            "check that the first template argument to the partitioner is a supported cell_topology type");
        /**@brief constructor

           suppose we are using an MPI cartesian communicator:
           then we have a coordinates frame (e.g. the local i,j,k identifying a processor id) and dimensions array (e.g.
         IxJxK).
           The constructor assigns the boundary flag in case the partition is at the boundary of the processors grid,
           regardless whether the communicator is periodic or not in that direction. The boundary assignment follows the
         convention
           represented in the figure below

            \verbatim
            ######2^0#######
            #              #
            #              #
            #              #
         2^(d+1)          2^1
            #              #
            #              #
            #              #
            #####2^(d)######
            \endverbatim

            where d is the number of dimensions of the processors grid.
            The boundary flag is a single integer containing the sum of the touched boundaries (i.e. a bit map).
        */
        partitioner_trivial(const communicator_t &comm,
            const gridtools::array< ushort_t, space_dimensions > &halo,
            const gridtools::array< ushort_t, space_dimensions > &padding)
            : m_pid(&comm.coordinates()[0]), m_ntasks(&comm.dimensions()[0]), m_halo(&halo[0]), m_pad(&padding[0]),
              m_comm(comm) {

            m_boundary = 0; // bitmap

            for (ushort_t i = 0; i < communicator_t::ndims; ++i)
                if (comm.coordinates(i) == comm.dimensions(i) - 1)
                    m_boundary += std::pow(2, i);
            for (ushort_t i = communicator_t::ndims; i < 2 * (communicator_t::ndims); ++i)
                if (comm.coordinates(i % (communicator_t::ndims)) == 0)
                    m_boundary += std::pow(2, i);
        }

        partitioner_trivial(const communicator_t &comm)
            : m_pid(comm.coordinates()), m_ntasks(&comm.dimensions()[0]), m_halo(NULL), m_pad(NULL), m_comm(comm) {

            m_boundary = 0; // bitmap

            for (ushort_t i = 0; i < communicator_t::ndims; ++i)
                if (comm.coordinates(i) == comm.dimensions(i) - 1)
                    m_boundary += std::pow(2, i);
            for (ushort_t i = communicator_t::ndims; i < 2 * (communicator_t::ndims); ++i)
                if (comm.coordinates(i % (communicator_t::ndims)) == 0)
                    m_boundary += std::pow(2, i);
        }

        void low_up_bounds(int_t &low_bound, int_t &up_bound, ushort_t component, uint_t const size_) const {
            if (component >= communicator_t::ndims || m_pid[component] == 0)
                low_bound = 0;
            else {
                div_t value = std::div(size_, m_ntasks[component]);
                low_bound = ((int)(value.quot * (m_pid[component])) +
                             (int)((value.rem >= m_pid[component]) ? (m_pid[component]) : value.rem));
/*error in the partitioner*/
// assert(low_bound[component]>=0);
#ifndef NDEBUG
                if (low_bound < 0) {
                    printf("\n\n\n ERROR[%d]: low bound for component %d is %d<0\n\n\n", PID, component, low_bound);
                }
#endif
            }

            if (component >= communicator_t::ndims || m_pid[component] == m_ntasks[component] - 1)
                up_bound = size_;
            else {
                div_t value = std::div(size_, m_ntasks[component]);
                up_bound = ((int)(value.quot * (m_pid[component] + 1)) +
                            (int)((value.rem > m_pid[component]) ? (m_pid[component] + 1) : value.rem));
                /*error in the partitioner*/
                // assert(up_bound[component]<size);
                if (up_bound > size_) {
                    printf(
                        "\n\n\n ERROR[%d]: up bound for component %d is %d>%d\n\n\n", PID, component, up_bound, size_);
                }
            }
        }

        uint_t compute_tile(ushort_t component, uint_t const size_) const {
            int_t low;
            int_t high;
            low_up_bounds(low, high, component, size_);
            return high - low;
        }

/**@brief computes the lower and upprt index of the local interval
   \param component the dimension being partitioned
   \param size the total size of the quantity being partitioned

   The bounds must be inclusive of the halo region
*/
#ifdef CXX11_ENABLED
        template < typename... UInt >
#endif
        uint_t compute_bounds(uint_t component,
            array< halo_descriptor, space_dimensions > &coordinates,
            array< halo_descriptor, space_dimensions > &coordinates_gcl,
            array< int_t, space_dimensions > &low_bound,
            array< int_t, space_dimensions > &up_bound,
#ifdef CXX11_ENABLED
            UInt const &... original_sizes
#else
            uint_t const &d1,
            uint_t const &d2,
            uint_t const &d3
#endif
            ) const {
#ifdef CXX11_ENABLED
            uint_t sizes[sizeof...(UInt)] = {original_sizes...};
            uint_t size_ = sizes[component];
#else
            uint_t sizes[3] = {d1, d2, d3};
            uint_t size_ = sizes[component];
#endif

            low_up_bounds(low_bound[component], up_bound[component], component, size_);

            uint_t tile_dimension = up_bound[component] - low_bound[component];

            coordinates[component] = halo_descriptor(compute_halo(component, LOW),
                compute_halo(component, UP),
                compute_halo(component, LOW),
                tile_dimension + (compute_halo(component, LOW)) - 1,
                tile_dimension + (compute_halo(component, UP)) + (compute_halo(component, LOW)));

            coordinates_gcl[component] = halo_descriptor(m_halo[component],
                m_halo[component],
                compute_halo(component, LOW),
                tile_dimension + (compute_halo(component, LOW)) - 1,
                tile_dimension + (compute_halo(component, UP)) + (compute_halo(component, LOW)));

#ifndef NDEBUG
#ifdef VERBOSE
            std::cout << "[" << PID << "]"
                      << "grid [" << compute_halo(component, LOW) << " " << compute_halo(component, UP) << " "
                      << compute_halo(component, LOW) << " " << tile_dimension + (compute_halo(component, LOW)) - 1
                      << " " << (tile_dimension + compute_halo(component, UP)) + (compute_halo(component, LOW)) << "]"
                      << std::endl;
            std::cout << "boundary for coords definition: " << boundary() << std::endl;
            std::cout << "partitioning" << std::endl;
            std::cout << "up bounds for component " << component << ": " << up_bound[component] << std::endl
                      << "low bounds for component " << component << ": " << low_bound[component] << std::endl
                      << "pid: " << m_pid[0] << " " << m_pid[1] << " " << m_pid[2] << std::endl
                      << "component, size: " << component << " " << size_ << std::endl;
#endif

#endif
            return tile_dimension + compute_halo(component, UP) + compute_halo(component, LOW);
        }

        /** @brief method to query the halo dimension based on the periodicity of the communicator and the boundary flag

            \param component is the current component, x or y (i.e. 0 or 1) in 2D
            \param flag is a flag identifying whether we are quering the high (flag=1) or low (flag=2^d) boundary,
            where d is the number of dimensions of the processor grid.
            Conventionally we consider the left/bottom corner as the frame origin, which is
            consistent with the domain indexing.

            Consider the following representation of the 2D domain with the given boundary flags
            \verbatim
            ####### 1 ######
            #              #
            #              #
            #              #
            8              2
            #              #
            #              #
            #              #
            ####### 4 ######
            \endverbatim

            example:
            - when component=0, flag=1, the grid is not periodic in x, and the boundary of this partition contains a
         portion of the "1" edge, this method returns 0.
            - when component=0, flag=4, the grid is not periodic in x, and the boundary of this partition contains a
         portion of the "4" edge, this method returns 0.
            - when component=1, flag=1, the grid is not periodic in y, and the boundary of this partition contains a
         portion of the "2" edge, this method returns 0.
            - when component=1, flag=4, the grid is not periodic in y, and the boundary of this partition contains a
         portion of the "8" edge, this method returns 0.
            - returns the halo in the component direction otherwise.

            The formula used for the computation is easily generalizable for hypercubes, given that
            the faces numeration satisfies a generalization of the following rule (where d is the dimension of the
         processors grid)

            \verbatim
            ######2^0#######
            #              #
            #              #
            #              #
         2^(d+1)          2^1
            #              #
            #              #
            #              #
            #####2^(d)######
            \endverbatim
        */
        int_t compute_halo(ushort_t const &component_, typename super::Flag const &flag_) const {
            return (m_comm.periodic(component_) || !at_boundary(component_, flag_)) ? m_halo[component_]
                                                                                    : m_pad[component_];
        }

        /**to be called from the user interface*/
        GT_FUNCTION
        bool at_boundary(ushort_t const &component_, typename super::Flag flag_) const {

            uint_t ret = (((uint_t)flag_ * (1<<component_)) ) & boundary();
            return !ret;
        }

        GT_FUNCTION
        uint_t boundary() const { return m_boundary; }

        template < int_t Component >
        int_t pid() const {
            return m_pid[Component];
        }

        void set_boundary(uint_t boundary_) { m_boundary = boundary_; }

      private:
        const int *m_pid;
        const int *m_ntasks;
        const ushort_t *m_halo;
        const ushort_t *m_pad;
        communicator_t const &m_comm;
        uint_t m_boundary;
    };

} // namespace gridtools
