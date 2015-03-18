#pragma once
#include "partitioner.h"
#include <common/halo_descriptor.h>

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

    //use of static polimorphism (partitioner methods may be accessed from whithin loops)
    template <typename Storage, typename Communicator>
    class partitioner_trivial : public partitioner<partitioner_trivial<Storage, Communicator> > {
    public:
        typedef Storage storage_t;
        typedef Communicator communicator_t;
        /**@brief constructor
           suppose we are using an MPI cartesian communicator:
           then we have a coordinates (e.g. the local i,j,k identifying a processor id) and dimensions (e.g. IxJxK)
        */
        partitioner_trivial(// communicator_t const& comm,
            const communicator_t& comm, const ushort_t* halo ): m_pid(comm.coordinates()), m_ntasks(comm.dimensions()), m_halo(halo), m_comm(comm){

            m_boundary=0;//bitmap
//             if(comm.coordinates(0)==comm.dimensions(0)-1) m_boundary  =  1; else m_boundary = 0;
//             if(comm.coordinates(1)==comm.dimensions(1)-1) m_boundary +=  2;
//             if(comm.coordinates(0)==0)  m_boundary += 4;
//             if(comm.coordinates(1)==0)  m_boundary += 8;

            //TODO think general
            for (ushort_t i=0; i<communicator_t::ndims-1; ++i)
                if(comm.coordinates(i)==comm.dimensions(i)-1) m_boundary  += std::pow(2, i);
            for (ushort_t i=communicator_t::ndims-1; i<2*(communicator_t::ndims-1); ++i)
                if(comm.coordinates(i-(communicator_t::ndims-1))==0) m_boundary  += std::pow(2, i);
        }


        virtual uint_t coord_length(ushort_t const& ID, uint_t const& size) const
            {
                uint_t ret;
                if(ID<communicator_t::ndims)
                {
                    div_t value=std::div(size, m_ntasks[ID]);
                    ret = ((int)(value.quot/**(pid+1)*/) + (int)((value.rem>m_pid[ID]) ? (/*pid+*/1) : value.rem));
                }
                else
                    ret = size;
                return ret;
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
                            UInt const& ... sizes
#else
                            uint_t const& d1, uint_t const& d2, uint_t const& d3
#endif
) const
            {
                //m_sizes[component]=size;
#ifdef CXX11_ENABLED
                uint_t sizes[sizeof...(UInt)]={sizes};
#else
                uint_t sizes[3]={d1, d2, d3};
#endif
                for(uint_t component=0; component<Storage::space_dimensions; ++component){
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
                            int pid=0;
                            MPI_Comm_rank(MPI_COMM_WORLD, &pid);

                            printf("\n\n\n ERROR[%d]: low bound for component %d is %d<0\n\n\n", pid,  component, low_bound[component] );
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
                            int pid=0;
                            MPI_Comm_rank(MPI_COMM_WORLD, &pid);
                            printf("\n\n\n ERROR[%d]: up bound for component %d is %d>%d\n\n\n", pid,  component, up_bound[component], sizes[component] );
                        }

                    }

                    uint_t tile_dimension = coord_length(component, sizes[component]);

                    //ushort_t fact=component+1;//TODO ugly and errorprone use enums/bitmaps
                    //                                            if component is periodic or if it is not on the border then add the halo
                    coordinates[component] = halo_descriptor( compute_halo(component,4),
                                                                compute_halo(component,1), //(m_comm.periodic(component) || m_comm.boundary()%(2*fact)<fact)?m_halo[component]:0,
                                                                compute_halo(component,4), //(m_comm.periodic(component) || m_comm.boundary()%(8*fact)<4*fact)?m_halo[component]:0,
                                                                tile_dimension + ( compute_halo(component,4)) - 1,
                                                                tile_dimension + ( compute_halo(component,1)) + (compute_halo(component,4)) ); //(m_comm.periodic(0) || m_comm.boundary()%(2*fact)<fact) ? m_halo[component]+1 : 1)

                    coordinates_gcl[component] = halo_descriptor( m_halo[component]-1,
                                                                m_halo[component]-1, //(m_comm.periodic(component) || m_comm.boundary()%(2*fact)<fact)?m_halo[component]:0,
                                                                compute_halo(component,4), //(m_comm.periodic(component) || m_comm.boundary()%(8*fact)<4*fact)?m_halo[component]:0,
                                                                tile_dimension + ( compute_halo(component,4)) - 1/*-1*/,//index error??
                                                                tile_dimension + ( compute_halo(component,1)) + (compute_halo(component,4)) ); //(m_comm.periodic(0) || m_comm.boundary()%(2*fact)<fact) ? m_halo[component]+1 : 1)

#ifndef NDEBUG
                int pid=0;
                MPI_Comm_rank(MPI_COMM_WORLD, &pid);

                std::cout<<"["<<pid<<"]"<<"coordinates"<< compute_halo(component,4)<<" "
                         <<compute_halo(component,1) << " "
                         << tile_dimension+(compute_halo(component,4))-1<<" "
                         <<( tile_dimension+ compute_halo(component,1))+(compute_halo(component,4))
                         << std::endl;
                std::cout<<"boundary for coords definition: "<<boundary()<<std::endl;
                std::cout<<"partitioning"<<std::endl;
                std::cout<<"up bounds: "<<up_bound[0]<<" "<<up_bound[1]<<" "<<up_bound[2]<<std::endl
                         <<"low bounds: "<<low_bound[0]<<" "<<low_bound[1]<<" "<<low_bound[2]<<std::endl
                         <<"pid: "<<m_pid[0]<<" "<<m_pid[1]<<" "<<m_pid[2]<<std::endl
                         <<"component, size: "<<component<<" "<<sizes[component]<<std::endl;
                printf("pid: %d, size: %d - %d + %d + %d\n", pid, up_bound[component], low_bound[component], compute_halo(component,1), compute_halo(component,4));
#endif
                dims[component]=up_bound[component]-low_bound[component]+ compute_halo(component,1)+compute_halo(component,4);
                }
            }

        int_t compute_halo(ushort_t const& component, ushort_t const& flag) const {
//             std::cout<<m_comm.periodic(component)<<" || "
//                      <<m_comm.boundary()<<" % 2*( "
//                      << component << "+1)*"<<flag
//                      << " <" <<((component+1)*flag)
//                      <<" ? "
//                      <<m_halo[component]<<" : 0 ====>"
//                      << (false || m_comm.boundary()%(2*(component+1)*flag)<((component+1)*flag)?m_halo[component]:0)
//                      << std::endl;
//TODO: how to query periodicity from m_comm?
            return (/*m_comm.periodic(component)*/false || boundary()%(2*(component+1)*flag)<((component+1)*flag))?m_halo[component]:0;
            }

        template<int_t Component>
        int_t pid() const {return m_pid[Component];}

//         GT_FUNCTION
//         communicator_t const& communicator() const {return m_comm;}

//         template<ushort_t dimension>
//         halo_descriptor const& get_halo_descriptor() const {return m_coordinates[dimension];}

//         template<ushort_t dimension>
//         halo_descriptor const& get_halo_gcl() const {return m_coordinates_gcl[dimension];}

//         template<uint_t Component>
//         uint_t const& global_offset() const {return m_low_bound[Component];}

        int const&  boundary()const{return m_boundary;}

    private:
        const int* m_pid;
        const int* m_ntasks;
        const ushort_t* m_halo;
        communicator_t const& m_comm;
        /**this are the offsets which allow to compute the global coordinates given the local ones*/
        //uint_t m_sizes[storage_t::space_dimensions];
        int m_boundary;
    };
}//namespace gridtools
