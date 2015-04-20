#pragma once
#include <boost/mpl/for_each.hpp>
#include "../backend_traits_fwd.h"

/**@file
@brief type definitions and structures specific for the CUDA backend*/
namespace gridtools{

    /**forward declaration*/
    namespace _impl_cuda{
        template <typename Arguments>
        struct run_functor_cuda;
    }

    /**forward declaration*/
    template<typename T>
    struct hybrid_pointer;

    /** @brief traits struct defining the types which are specific to the CUDA backend*/
    template<>
    struct backend_traits_from_id< enumtype::Cuda >
    {

        template <typename T>
        struct pointer
        {
            typedef hybrid_pointer<T> type;
        };

        template <typename ValueType, typename Layout, bool Temp=false, short_t SpaceDim=1 >
        struct storage_traits
        {
            typedef storage< base_storage<typename pointer<ValueType>::type, Layout, Temp, SpaceDim> > storage_t;
        };

        template <typename Arguments>
        struct execute_traits
        {
            typedef _impl_cuda::run_functor_cuda<Arguments> run_functor_t;
        };

        /** This is the function used by the specific backend to inform the
            generic backend and the temporary storage allocator how to
            compute the number of threads in the i-direction, in a 2D
            grid of threads.
        */
        GT_FUNCTION
        static uint_t n_i_pes(int = 0) {
            return gridDim.x;
       }

        /** This is the function used by the specific backend to inform the
            generic backend and the temporary storage allocator how to
            compute the number of threads in the j-direction, in a 2D
            grid of threads.
        */
        GT_FUNCTION
        static uint_t n_j_pes(int = 0) {
            return gridDim.y;
        }

        /** This is the function used by the specific backend
         *  that determines the i coordinate of a processing element.
         *  In the case of CUDA, a processing element is equivalent to a CUDA block
         */
        GT_FUNCTION
        static uint_t processing_element_i() {
            return blockIdx.x;
        }

        /** This is the function used by the specific backend
         *  that determines the j coordinate of a processing element.
         *  In the case of CUDA, a processing element is equivalent to a CUDA block
         */
        GT_FUNCTION
        static uint_t processing_element_j() {
            return blockIdx.y;
        }

#ifdef CXX11_ENABLED
        //function alias (pre C++11)
        template<typename Sequence, typename F>
        GT_FUNCTION
        static void for_each(F&& f)
            {
                boost::mpl::for_each<Sequence>(std::forward<F>(f));
            }
#else
        //function alias (pre C++11)
        template<typename Sequence, typename F>
        GT_FUNCTION
        static void for_each(F f)
            {
                boost::mpl::for_each<Sequence>(f);
            }
#endif

        /**
           @brief assigns the two given values using the given thread Id whithin the block
        */
        template <uint_t Id>
        struct once_per_block {
            template<typename Left, typename Right>
            GT_FUNCTION
            static void assign(Left& l, Right const& r){
                if(threadIdx.x==Id)
                    {
                        l=r;
                    }
            }
        };

    };

}//namespace gridtools
