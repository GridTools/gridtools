
#pragma once
#include <gt_for_each/for_each.hpp>


namespace gridtools{
    namespace _impl_host{
        template <typename Arguments>
        struct run_functor_host;
    }


    template <enumtype::backend BE, typename T, typename U, bool B>
    struct base_storage;

    // template <typename ValueType, typename Layout, bool Temp>
    // struct storage;

    template<typename T>
    struct wrap_pointer;


    template<enumtype::backend T>
    struct backend_from_id;

/**Traits struct, containing the types which are specific for the host backend*/
    template<>
    struct backend_from_id<enumtype::Host>
    {
        template <typename ValueType, typename Layout, bool Temp=false>
        struct storage_traits{
            typedef base_storage<enumtype::Host, ValueType, Layout, Temp> storage_t;
        };

        template <typename Arguments>
        struct execute_traits
        {
            typedef _impl_host::run_functor_host< Arguments > backend_t;
        };

        //function alias (pre C++11, std::bind or std::mem_fn,
        //using function pointers looks very ugly)
        template<
            typename Sequence
            , typename F
            >
        //unnecessary copies/indirections if the compiler is not smart (std::forward)
        inline static void for_each(F f)
            {
                gridtools::for_each<Sequence>(f);
            }

        template <typename T>
        inline static void delete_storage(wrap_pointer<T>& data){ data.free_it();/*delete[] &data[0];*/}

        template <typename T>
        struct pointer
        {
            typedef wrap_pointer<T> type;
        };

        GT_FUNCTION
        static void assertion(bool const condition) {
            assert(condition);
        }

    };
}//namespace gridtools
