
#pragma once
#include <gt_for_each/for_each.hpp>

/**@file
@brief type definitions and structures specific for the Host backend*/

namespace gridtools{
    namespace _impl_host{
	/**forward declaration*/
        template <typename Arguments>
        struct run_functor_host;
    }


    /**forward declaration*/
    template <enumtype::backend BE, typename T, typename U, bool B, short_t SpaceDim>
    struct base_storage;

    // /**forward declaration*/
    // template <typename U>
    // struct storage;

    /**forward declaration*/
    template<typename T>
    struct wrap_pointer;

    /**forward declaration*/
    template<enumtype::backend T>
    struct backend_from_id;

/**Traits struct, containing the types which are specific for the host backend*/
    template<>
    struct backend_from_id<enumtype::Host>{
        template <typename ValueType, typename Layout, bool Temp=false, short_t SpaceDim=1>
        struct storage_traits{
            typedef base_storage<enumtype::Host, ValueType, Layout, Temp, SpaceDim >   storage_t;
        };

        template <typename Arguments>
        struct execute_traits{
            typedef _impl_host::run_functor_host< Arguments > backend_t;
        };

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

        template <typename T>
        struct pointer
        {
            typedef wrap_pointer<T> type;
        };

    };

    /**forward declaration*/
    template <enumtype::backend, uint_t Id>
    struct once_per_block;

    template <uint_t Id>
    struct once_per_block<enumtype::Host, Id>{
	template<typename Left, typename Right>
	GT_FUNCTION//inline
	static void assign(Left& l, Right const& r){
	    l=r;
	}
    };
}//namespace gridtools
