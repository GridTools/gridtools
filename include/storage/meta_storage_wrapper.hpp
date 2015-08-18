#pragma once
#include "meta_storage.hpp"
#include "meta_storage_tmp.hpp"

namespace gridtools{
template < typename BaseStorage >
struct meta_storage_wrapper : public BaseStorage, clonable_to_gpu<meta_storage_wrapper<BaseStorage> >
    {
        static const bool is_temporary=BaseStorage::is_temporary;
        typedef BaseStorage super;
        typedef typename BaseStorage::basic_type basic_type;
        typedef typename BaseStorage::index_type index_type;
        typedef meta_storage_wrapper<BaseStorage> original_storage;
        typedef clonable_to_gpu<meta_storage_wrapper<BaseStorage> > gpu_clone;

        __device__
        meta_storage_wrapper(BaseStorage const& other)
            :  super(other)
            {}

#if defined(CXX11_ENABLED)
        //arbitrary dimensional field
        template <class ... UIntTypes>
        explicit meta_storage_wrapper(  UIntTypes const& ... args ): super(args ...)
            {
            }
#else
        //constructor picked in absence of CXX11 or which GCC<4.9
        explicit meta_storage_wrapper(uint_t const& dim1, uint_t const& dim2, uint_t const& dim3): super(dim1, dim2, dim3) {}
#endif



//    private :
        explicit meta_storage_wrapper(): super(){}
        // BaseStorage m_meta_storage;

    };


#ifdef CXX11_ENABLED
    template < ushort_t Index
               , typename Layout
               , bool IsTemporary
               , typename ... Tiles
               >
    using meta_storage = meta_storage_wrapper<meta_storage_base<Index, Layout, IsTemporary, Tiles...> >;
#else
    template < ushort_t Index
               , typename Layout
               , bool IsTemporary
               , typename ... Tiles
               >
    struct meta_storage<> : meta_storage_wrapper<meta_storage_base<Index, Layout, IsTemporary, Tiles...> >{};
#endif

    template<typename T>
    struct is_meta_storage : boost::mpl::false_{};

    template< typename Storage>
    struct is_meta_storage<meta_storage_wrapper<Storage> > : boost::mpl::true_{};

    template < ushort_t Index
               , typename Layout
               , bool IsTemporary
               , typename ... Tiles
               >
    struct is_meta_storage<meta_storage<Index, Layout, IsTemporary, Tiles...> > : boost::mpl::true_{};

    template<ushort_t Index, typename Layout, bool IsTemporary, typename ... Whatever>
    struct is_meta_storage<meta_storage_base<Index, Layout, IsTemporary, Whatever...> > : boost::mpl::true_{};

    template<typename T>
    struct is_meta_storage_wrapper : is_meta_storage<typename boost::remove_pointer<T>::type::super>{};


    template<typename T>
    struct is_ptr_to_meta_storage_wrapper : boost::mpl::false_ {};

    template<typename T>
    struct is_ptr_to_meta_storage_wrapper<pointer<const T> > : is_meta_storage_wrapper<T> {};


} // namespace gridtools
