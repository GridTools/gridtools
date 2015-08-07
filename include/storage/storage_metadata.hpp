#pragma once

namespace gridtools{
template < typename BaseStorage >
struct meta_storage_wrapper : public BaseStorage, clonable_to_gpu<meta_storage_wrapper<BaseStorage> >
    {
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
} // namespace gridtools
