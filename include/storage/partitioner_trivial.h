#pragma once
#include "partitioner.h"
#include <common/halo_descriptor.h>

/**
@file
@brief Simple Partitioner Class
This file defines a simple partitioner splitting a structured cartesian grid

Hypotheses:
* tensor product grid structure
* halo region is the same on both sizes in one direction, but can differ in every direction
* there is no padding, i.e. the first address index "begin" coincides with the halo in that direction. Same for the last address index.
*/

namespace gridtools{

    //use of static polimorphism (partitioner methods may be accessed from whithin loops)
    template <typename Storage>
    class partitioner_trivial : public partitioner<partitioner_trivial<Storage> > {
    public:
        typedef Storage storage_t;
        /**@brief constructor
           suppose we are using an MPI cartesian communicator:
           then we have a coordinates (e.g. the local i,j,k identifying a processor id) and dimensions (e.g. IxJxK)
        */
        partitioner_trivial(int size, int* coordinates, int* dims, ushort_t* halo): m_partition_size(size), m_pid(coordinates), m_ntasks(dims), m_halo(halo){
        }

        /**returns the sizes of a specific dimension in the current partition*/
        virtual uint_t* sizes() {
            return m_sizes;
        }

        virtual uint_t coord_length(ushort_t const& ID)
            {
                div_t value=std::div(m_sizes[ID]-m_halo[ID]-m_halo[ID],m_ntasks[ID]);
                return ((int)(value.quot/**(pid+1)*/) + (int)((value.rem>m_pid[ID]) ? (/*pid+*/1) : value.rem));
            }

        /**@brief computes the lower and upprt index of the local interval
           \param component the dimension being partitioned
           \param size the total size of the quantity being partitioned

           The bounds must be inclusive of the halo region
        */
        virtual uint_t compute_bounds(ushort_t const& component, uint_t const&size)
            {
                m_sizes[component]=size;
                //
                if (m_pid[component]==0)
                    m_low_bound[component] = 0;
                else
                {
                    div_t value=std::div(size,m_ntasks[component]);
                    m_low_bound[component] = ((int)(value.quot*(m_pid[component])) + (int)((value.rem>=m_pid[component]) ? (m_pid[component]) : value.rem))-m_halo[component];
                    /*error in the partitioner*/
                    assert(m_low_bound[component]>=0);
                }

                if (m_pid[component]==m_ntasks[component]-1)
                    m_up_bound[component] = size-1;
                else
                {
                    div_t value=std::div(size,m_ntasks[component]);
                    m_up_bound[component] = ((int)(value.quot*(m_pid[component]+1)) + (int)((value.rem>m_pid[component]) ? (m_pid[component]+1) : value.rem));
                    /*error in the partitioner*/
                    assert(m_up_bound[component]<size);
                }

                uint_t tile_dimension = coord_length(component);
                m_coordinates[component] = halo_descriptor(m_halo[component], m_halo[component], m_halo[component], tile_dimension, tile_dimension+m_halo[component]+1);

#ifndef NDEBUG
                std::cout<<"partitioning"<<std::endl;
                std::cout<<"up bounds: "<<m_up_bound[0]<<" "<<m_up_bound[1]<<" "<<m_up_bound[2]<<std::endl
                         <<"low bounds: "<<m_low_bound[0]<<" "<<m_low_bound[1]<<" "<<m_low_bound[2]<<std::endl
                         <<"pid: "<<m_pid[0]<<" "<<m_pid[1]<<" "<<m_pid[2]<<std::endl
                         <<"component, size: "<<component<<" "<<size<<std::endl;
#endif
                return m_up_bound[component]-m_low_bound[component]+m_halo[component];
            }

        template<int_t Component>
        int_t pid(){return m_pid[Component];}

        template<ushort_t dimension>
        halo_descriptor const& get_halo_descriptor(){return m_coordinates[dimension];}

        template<uint_t Component>
        uint_t const& global_offset(){return m_low_bound[Component];}

    private:
        uint_t m_partition_size;
        int* m_pid;
        int* m_ntasks;
        uint_t* m_halo;
        /**this are the offsets which allow to compute the global coordinates given the local ones*/
        int_t m_low_bound[Storage::space_dimensions];
        int_t m_up_bound[Storage::space_dimensions];
        uint_t m_sizes[Storage::space_dimensions];
        halo_descriptor m_coordinates[Storage::space_dimensions];
    };
}//namespace gridtools
