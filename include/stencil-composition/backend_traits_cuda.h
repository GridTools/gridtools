#pragma once
#include <boost/mpl/for_each.hpp>

namespace gridtools{

    namespace _impl_cuda{
        template <typename Arguments>
        struct run_functor_cuda;
    }

    template <enumtype::backend BE, typename T, typename U, bool B, short_t SpaceDim>
    struct base_storage;

    template <typename U>
      struct storage;

    // template <typename ValueType, typename Layout, bool Temp>
    // struct cuda_storage;

    template<typename T>
    struct hybrid_pointer;

    template<enumtype::backend T>
    struct backend_from_id;

/** traits struct defining the types which are specific to the CUDA backend*/
    template<>
    struct backend_from_id< enumtype::Cuda >
    {

        template <typename ValueType, typename Layout, bool Temp=false, short_t SpaceDim=1 >
        struct storage_traits
        {
            typedef storage< base_storage<enumtype::Cuda, ValueType, Layout, Temp, SpaceDim> > storage_t;
        };

        template <typename Arguments>
        struct execute_traits
        {
            typedef _impl_cuda::run_functor_cuda<Arguments> backend_t;
        };

        //function alias (pre C++11)
        template<
            typename Sequence
            , typename F
            >
        inline static void for_each(F f)
            {
                boost::mpl::for_each<Sequence>(f);
            }

        template <typename T>
	  inline static void delete_storage(hybrid_pointer<T>& data){ data.free_it();}

        template <typename T>
        struct pointer
        {
            typedef hybrid_pointer<T> type;
        };

    };
}//namespace gridtools
