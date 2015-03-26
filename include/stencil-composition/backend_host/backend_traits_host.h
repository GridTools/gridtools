#pragma once
#include <gt_for_each/for_each.hpp>
#include "../backend_traits_fwd.h"

/**@file
@brief type definitions and structures specific for the Host backend*/

namespace gridtools{
    namespace _impl_host{
        /**forward declaration*/
        template <typename Arguments>
        struct run_functor_host;
    }


    /**forward declaration*/
    template<typename T>
    struct wrap_pointer;

    /**Traits struct, containing the types which are specific for the host backend*/
    template<>
    struct backend_traits_from_id<enumtype::Host>{

        template <typename T>
        struct pointer
        {
            typedef wrap_pointer<T> type;
        };

        template <typename ValueType, typename Layout, bool Temp=false, short_t FieldDim=1>
        struct storage_traits{
            typedef storage<base_storage<typename pointer<ValueType>::type, Layout, Temp, FieldDim > >   storage_t;
        };

        template <typename Arguments>
        struct execute_traits{
            typedef _impl_host::run_functor_host< Arguments > run_functor_t;
        };

        /** This is the function used by the specific backend to inform the
            generic backend and the temporary storage allocator how to
            compute the number of threads in the i-direction, in a 2D
            grid of threads.
        */
        static uint_t n_i_threads(int = 0) {
            return n_threads();
        }

        /** This is the function used by the specific backend to inform the
            generic backend and the temporary storage allocator how to
            compute the number of threads in the j-direction, in a 2D
            grid of threads.
        */
        static uint_t n_j_threads(int = 0) {
            return 1;
        }

        //function alias (pre C++11, std::bind or std::mem_fn,
        //using function pointers looks very ugly)
        template<
            typename Sequence
            , typename F
            >
        //unnecessary copies/indirections if the compiler is not smart (std::forward)
        inline static void for_each(F f){
                gridtools::for_each<Sequence>(f);
            }

        template <uint_t Id>
        struct once_per_block {
            template<typename Left, typename Right>
            GT_FUNCTION//inline
            static void assign(Left& l, Right const& r){
                l=r;
            }
        };
    };

}//namespace gridtools
