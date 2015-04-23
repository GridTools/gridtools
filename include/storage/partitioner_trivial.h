#pragma once
#include "partitioner.h"

/**
@file
@brief Simple Partitioner Class
This file defines a simple partitioner splitting a structured cartesian grid

Hypotheses:
- tensor product grid structure
- halo region is the same on both sizes in one direction, but can differ in every direction
- there is no padding, i.e. the first address index "begin" coincides with the halo in that direction. Same for the last address index.

The partitioner class is storage-agnostic, does not know anything about the storage, and thus the same partitioner can be applied to multiple storages
*/

namespace gridtools{

    template<typename TopologyType>
    class cell_topology;

    template <typename T>
    struct is_cell_topology : boost::false_type {};

    template <typename TopologyType>
    struct is_cell_topology<cell_topology<TopologyType> > : boost::true_type {};

    //use of static polimorphism (partitioner methods may be accessed from whithin loops)
    template <typename GridTopology, typename Communicator>
    class partitioner_trivial : public partitioner<partitioner_trivial<GridTopology, Communicator> > {
    public:
        typedef GridTopology topology_t;
        typedef Communicator communicator_t;

        GRIDTOOLS_STATIC_ASSERT(is_cell_topology<GridTopology>::value, "check that the first template argument to the partitioner is a supported cell_topology type")
        /**@brief constructor

           suppose we are using an MPI cartesian communicator:
           then we have a coordinates frame (e.g. the local i,j,k identifying a processor id) and dimensions array (e.g. IxJxK).
           The constructor assigns the boundary flag in case the partition is at the boundary of the processors grid,
           regardless whether the communicator is periodic or not in that direction. The boundary assignment follows the convention
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
        partitioner_trivial(// communicator_t const& comm,
            const communicator_t& comm, const gridtools::array<ushort_t, topology_t::space_dimensions>& halo )
            : m_pid(comm.coordinates()), m_ntasks(&comm.dimensions()[0]), m_halo(&halo[0]), m_comm(comm){

            m_boundary=0;//bitmap

            for (ushort_t i=0; i<communicator_t::ndims; ++i)
                if(comm.coordinates(i)==comm.dimensions(i)-1) m_boundary  += std::pow(2, i);
            for (ushort_t i=communicator_t::ndims; i<2*(communicator_t::ndims); ++i)
                if(comm.coordinates(i%(communicator_t::ndims))==0) m_boundary  += std::pow(2, i);
        }


        /**@brief computes the lower and upprt index of the local interval
           \param component the dimension being partitioned
           \param size the total size of the quantity being partitioned

           The bounds must be inclusive of the halo region
        */
#ifdef CXX11_ENABLED
        template<typename ... UInt>
#endif
        void compute_bounds(uint_t* dims, halo_descriptor * coordinates, halo_descriptor * coordinates_gcl, int_t* low_bound, int_t* up_bound,
#ifdef CXX11_ENABLED
                            UInt const& ... original_sizes
#else
                            uint_t const& d1, uint_t const& d2, uint_t const& d3
#endif
) const
            {
                //m_sizes[component]=size;
#ifdef CXX11_ENABLED
                uint_t sizes[sizeof...(UInt)]={original_sizes...};
#else
                uint_t sizes[3]={d1, d2, d3};
#endif
                for(uint_t component=0; component<topology_t::space_dimensions; ++component){
                    if ( component >= communicator_t::ndims || m_pid[component]==0 )
                        low_bound[component] = 0;
                    else
                    {
                        div_t value=std::div(sizes[component],m_ntasks[component]);
                        low_bound[component] = ((int)(value.quot*(m_pid[component])) + (int)((value.rem>=m_pid[component]) ? (m_pid[component]) : value.rem));
                        /*error in the partitioner*/
                        //assert(low_bound[component]>=0);
#ifndef NDEBUG
                        if(low_bound[component]<0){
                            printf("\n\n\n ERROR[%d]: low bound for component %d is %d<0\n\n\n", PID,  component, low_bound[component] );
                        }
#endif
                    }

                    if (component >= communicator_t::ndims || m_pid[component]==m_ntasks[component]-1 )
                        up_bound[component] = sizes[component];
                    else
                    {
                        div_t value=std::div(sizes[component],m_ntasks[component]);
                        up_bound[component] = ((int)(value.quot*(m_pid[component]+1)) + (int)((value.rem>m_pid[component]) ? (m_pid[component]+1) : value.rem));
                        /*error in the partitioner*/
                        //assert(up_bound[component]<size);
                        if(up_bound[component]>sizes[component]){
                            printf("\n\n\n ERROR[%d]: up bound for component %d is %d>%d\n\n\n", PID,  component, up_bound[component], sizes[component] );
                        }

                    }

                    uint_t tile_dimension = up_bound[component]-low_bound[component];

                    coordinates[component] = halo_descriptor( compute_halo(component,8),
                                                                compute_halo(component,1),
                                                                compute_halo(component,8),
                                                                tile_dimension + ( compute_halo(component,8)) - 1,
                                                                tile_dimension + ( compute_halo(component,1)) + (compute_halo(component,8)) );

                    coordinates_gcl[component] = halo_descriptor( m_halo[component],
                                                                  m_halo[component],
                                                                  compute_halo(component,8),
                                                                  tile_dimension + ( compute_halo(component,8)) - 1,
                                                                  tile_dimension + ( compute_halo(component,1)) + (compute_halo(component,8)) );

#ifndef NDEBUG
                std::cout<<"["<<PID<<"]"<<"coordinates ["<< compute_halo(component,8)<<" "
                         <<compute_halo(component,1) << " "
                         <<compute_halo(component,8) << " "
                         << tile_dimension+(compute_halo(component,8))-1<<" "
                         <<( tile_dimension+ compute_halo(component,1))+(compute_halo(component,8))<<"]"
                         << std::endl;
                std::cout<<"boundary for coords definition: "<<boundary()<<std::endl;
                std::cout<<"partitioning"<<std::endl;
                std::cout<<"up bounds for component "<< component <<": "<<up_bound[component]<<std::endl
                         <<"low bounds for component "<< component <<": "<<low_bound[component]<<std::endl
                         <<"pid: "<<m_pid[0]<<" "<<m_pid[1]<<" "<<m_pid[2]<<std::endl
                         <<"component, size: "<<component<<" "<<sizes[component]<<std::endl;
#endif
                dims[component]=up_bound[component]-low_bound[component]+ compute_halo(component,1)+compute_halo(component,8);
                }
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
            - when component=0, flag=1, the grid is not periodic in x, and the boundary of this partition contains a portion of the "1" edge, this method returns 0.
            - when component=0, flag=4, the grid is not periodic in x, and the boundary of this partition contains a portion of the "4" edge, this method returns 0.
            - when component=1, flag=1, the grid is not periodic in y, and the boundary of this partition contains a portion of the "2" edge, this method returns 0.
            - when component=1, flag=4, the grid is not periodic in y, and the boundary of this partition contains a portion of the "8" edge, this method returns 0.
            - returns the halo in the component direction otherwise.

            The formula used for the computation is easily generalizable for hypercubes, given that
            the faces numeration satisfies a generalization of the following rule (where d is the dimension of the processors grid)

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
        int_t compute_halo(ushort_t const& component, ushort_t const& flag) const {
            return (  m_comm.periodic(component) || boundary()%((ushort_t)std::pow(2,component+1)*flag)<((component+1)*flag))?m_halo[component]:0;
            }

        template<int_t Component>
        int_t pid() const {return m_pid[Component];}

        int const&  boundary()const{return m_boundary;}

    private:
        const int* m_pid;
        const int* m_ntasks;
        const ushort_t* m_halo;
        communicator_t const& m_comm;
        int m_boundary;
    };
}//namespace gridtools
