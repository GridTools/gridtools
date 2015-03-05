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
            if(comm.coordinates(0)==comm.dimensions(0)-1) m_boundary  =  1; else m_boundary = 0;
            if(comm.coordinates(1)==comm.dimensions(1)-1) m_boundary +=  2;
            if(comm.coordinates(0)==0)  m_boundary += 4;
            if(comm.coordinates(1)==0)  m_boundary += 8;
//             for (ushort_t i=0; i</*communicator_t::ndims*/2; ++i)
//                 if(comm.coordinates(i)==comm.dimensions(i)-1) m_boundary  += std::pow(2, i);
//             for (ushort_t i=/*communicator_t::ndims*/2; i<2*/*communicator_t::ndims*/2; ++i)
//                 if(comm.coordinates(i-/*communicator_t::ndims*/2)==0) m_boundary  += std::pow(2, i);

        }

        /**returns the sizes of a specific dimension in the current partition*/
        virtual uint_t* sizes() {
            return m_sizes;
        }

        virtual uint_t coord_length(ushort_t const& ID)
            {
                uint_t ret;
                if(ID<communicator_t::ndims)
                {
                    div_t value=std::div(m_sizes[ID],m_ntasks[ID]);
                    ret = ((int)(value.quot/**(pid+1)*/) + (int)((value.rem>m_pid[ID]) ? (/*pid+*/1) : value.rem));
                }
                else
                    ret = m_sizes[ID];
                return ret;
            }

        /**@brief computes the lower and upprt index of the local interval
           \param component the dimension being partitioned
           \param size the total size of the quantity being partitioned

           The bounds must be inclusive of the halo region
        */
        virtual uint_t compute_bounds(ushort_t const& component, uint_t const&size)
            {
                m_sizes[component]=size;

                    if ( component >= communicator_t::ndims || m_pid[component]==0 )
                        m_low_bound[component] = 0;
                    else
                    {
                        div_t value=std::div(size,m_ntasks[component]);
                        m_low_bound[component] = ((int)(value.quot*(m_pid[component])) + (int)((value.rem>=m_pid[component]) ? (m_pid[component]) : value.rem));
                    /*error in the partitioner*/
                    //assert(m_low_bound[component]>=0);
#ifndef NDEBUG
                        if(m_low_bound[component]<0){
                            int pid=0;
                            MPI_Comm_rank(MPI_COMM_WORLD, &pid);

                            printf("\n\n\n ERROR[%d]: low bound for component %d is %d<0\n\n\n", pid,  component, m_low_bound[component] );
                        }
#endif
                    }

                    if (component >= communicator_t::ndims || m_pid[component]==m_ntasks[component]-1 )
                        m_up_bound[component] = size;
                    else
                    {
                        div_t value=std::div(size,m_ntasks[component]);
                        m_up_bound[component] = ((int)(value.quot*(m_pid[component]+1)) + (int)((value.rem>m_pid[component]) ? (m_pid[component]+1) : value.rem));
                        /*error in the partitioner*/
                    //assert(m_up_bound[component]<size);
                        if(m_up_bound[component]>size){
                            int pid=0;
                            MPI_Comm_rank(MPI_COMM_WORLD, &pid);
                            printf("\n\n\n ERROR[%d]: up bound for component %d is %d>%d\n\n\n", pid,  component, m_up_bound[component], size );
                        }

                    }

                uint_t tile_dimension = coord_length(component);

                    //ushort_t fact=component+1;//TODO ugly and errorprone use enums/bitmaps
                    //                                            if component is periodic or if it is not on the border then add the halo
                    m_coordinates[component] = halo_descriptor( compute_halo(component,4),
                                                                compute_halo(component,1), //(m_comm.periodic(component) || m_comm.boundary()%(2*fact)<fact)?m_halo[component]:0,
                                                                compute_halo(component,4), //(m_comm.periodic(component) || m_comm.boundary()%(8*fact)<4*fact)?m_halo[component]:0,
                                                                tile_dimension + ( compute_halo(component,4)) - 1,
                                                                tile_dimension + ( compute_halo(component,1)) + (compute_halo(component,4)) ); //(m_comm.periodic(0) || m_comm.boundary()%(2*fact)<fact) ? m_halo[component]+1 : 1)

                    m_coordinates_gcl[component] = halo_descriptor( m_halo[component]-1,
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
                std::cout<<"up bounds: "<<m_up_bound[0]<<" "<<m_up_bound[1]<<" "<<m_up_bound[2]<<std::endl
                         <<"low bounds: "<<m_low_bound[0]<<" "<<m_low_bound[1]<<" "<<m_low_bound[2]<<std::endl
                         <<"pid: "<<m_pid[0]<<" "<<m_pid[1]<<" "<<m_pid[2]<<std::endl
                         <<"component, size: "<<component<<" "<<size<<std::endl;
                printf("pid: %d, size: %d - %d + %d + %d\n", pid, m_up_bound[component], m_low_bound[component], compute_halo(component,1), compute_halo(component,4));
#endif
                return m_up_bound[component]-m_low_bound[component]+ compute_halo(component,1)+compute_halo(component,4);
            }

        int_t compute_halo(ushort_t const& component, ushort_t const& flag){
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

        template<ushort_t dimension>
        halo_descriptor const& get_halo_descriptor() const {return m_coordinates[dimension];}

        template<ushort_t dimension>
        halo_descriptor const& get_halo_gcl() const {return m_coordinates_gcl[dimension];}

        template<uint_t Component>
        uint_t const& global_offset() const {return m_low_bound[Component];}

        int const&  boundary()const{return m_boundary;}

    private:
        const int* m_pid;
        const int* m_ntasks;
        const ushort_t* m_halo;
        communicator_t const& m_comm;
        /**this are the offsets which allow to compute the global coordinates given the local ones*/
        int_t m_low_bound[Storage::space_dimensions];
        int_t m_up_bound[Storage::space_dimensions];
        uint_t m_sizes[Storage::space_dimensions];
        halo_descriptor m_coordinates[Storage::space_dimensions];
        halo_descriptor m_coordinates_gcl[Storage::space_dimensions];
        int m_boundary;
    };
}//namespace gridtools
