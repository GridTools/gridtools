#pragma once
#include "partitioner.h"

/**
@file
@brief Simple Partitioner Class
This file defines a simple partitioner splitting a structured cartesian grid
*/
template <typename Storage>
class partitioner_trivial : public partitioner{

public:
    /**@brief constructor
       suppose we are using an MPI cartesian communicator:
       then we have a coordinates (e.g. the local i,j,k identifying a processor id) and dimensions (e.g. IxJxK)
    */
    partitioner_trivial(int* coordinates, int* dims, ushort_t* halo): m_pid(coordinates), m_ntasks(dims), m_halo(halo){
    }

    // template<std::size_t id>
    // struct get_pack{
    //     template<typename First, typename ... UInt>
    //     static ushort_t apply( First first, UInt ... sequence){
    //         return sizeof...(UInt)==id? first : apply(sequence);
    //     }
    // }

    // template<ushort_t N, typename Sequence>
    // struct assign_sizes{
    //     static const ushort_t id=boost::mpl::at_c<Sequence, N>::type::value;

    //     template<typename ... UInt>
    //     static void apply(uint_t* sizes, std::tuple<UInt ...> const& dims ){
    //         sizes[N]=up_bound(std::get<id>(dims))-low_bound(std::get<id>(dims));
    //         assign_sizes<N-1, Sequence>::apply(dims);
    //     }
    // };

    // template<typename Sequence>
    // struct assign_sizes<0, Sequence>{
    //     static const ushort_t id=boost::mpl::at_c<Sequence, 0>::type::value;

    //     template<typename ... UInt>
    //     static void apply(uint_t* sizes, std::tuple<UInt ...> const& dims ){
    //         sizes[0]=up_bound(std::get<id>(dims))-low_bound(std::get<id>(dims));
    //     }
    // };

    // template<typename ... UInt>
    // void compute_partition(UInt const& ... dims){
    //     using sequence = typename boost::mpl::range_c<ushort_t, 0, sizeof...(dims)>::type;
    //     //GRIDTOOLS_ASSERT(sizeof...(dims)==Storage::space_dimensions);
    //     const std::tuple<UInt ...> dimensions(dims ...);
    //     assign_sizes<Storage::space_dimensions, sequence>::apply(m_sizes, dimensions);
    // }

    // /**returns the sizes of a specific dimension in the current partition*/
    // virtual uint_t size(ushort_t const& component) const {
    //     return up_bound(std::get<id>(dims))-low_bound(std::get<id>(dims));
    // }

    /**returns the sizes of a specific dimension in the current partition*/
    virtual uint_t* sizes() {
        return m_sizes;
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
            std::cout<<"partitioning"<<std::endl;
            std::cout<<"up bounds: "<<m_up_bound[0]<<" "<<m_up_bound[1]<<" "<<m_up_bound[2]<<std::endl
                     <<"low bounds: "<<m_low_bound[0]<<" "<<m_low_bound[1]<<" "<<m_low_bound[2]<<std::endl
                     <<"pid: "<<m_pid[0]<<" "<<m_pid[1]<<" "<<m_pid[2]<<std::endl
                     <<"component, size: "<<component<<" "<<size<<std::endl;
            return m_up_bound[component]-m_low_bound[component]+m_halo[component];
        }
private:
    int_t m_low_bound[Storage::space_dimensions];
    int_t m_up_bound[Storage::space_dimensions];
    uint_t m_sizes[Storage::space_dimensions];
    ushort_t* m_halo;
    int* m_pid;
    int* m_ntasks;
};
