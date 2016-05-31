#pragma once
#include "../../storage/meta_storage_base.hpp"
#include "../../common/generic_metafunctions/unzip.hpp"

namespace gridtools{

#if !defined( __CUDACC__) || defined( CUDA8 )

    template <typename Layout, uint_t ... Dims>
    struct meta_storage_cache{

        typedef meta_storage_base<0, Layout, false > meta_storage_t;

    private:
        const meta_storage_t m_value;
    public:

        GT_FUNCTION
        constexpr meta_storage_cache( meta_storage_cache const& other) : m_value{other.m_value}{};

        GT_FUNCTION
        constexpr meta_storage_cache() : m_value{Dims...}{};

        GT_FUNCTION
        constexpr const meta_storage_t value() {return m_value;}

        GT_FUNCTION
        constexpr uint_t const& size() {return m_value.size();}

        template <typename Accessor>
        GT_FUNCTION
        constexpr const int_t index(Accessor const& arg_) {return  m_value._index(arg_);}
    };
#else

    template <typename Layout, uint_t Dim1, uint_t Dim2, uint_t Dim3, uint_t FD>
    struct meta_storage_cache{

    public:

        GT_FUNCTION
        constexpr meta_storage_cache(){};

        GT_FUNCTION
        constexpr uint_t const& size() {return Dim1*Dim2*FD;}

        template <typename Accessor>
        GT_FUNCTION
        // constexpr
        const int_t index(Accessor const& arg_) const {
            if(Accessor::n_dim>3)
                return (arg_.template get< Accessor::n_dim - 1 >() +
                        (arg_.template get< Accessor::n_dim - 2 >()) * Dim1  +
                        arg_.template get< 0 >()*Dim1*Dim2);
            else
                return (arg_.template get< Accessor::n_dim - 1 >() +
                        (arg_.template get< Accessor::n_dim - 2 >()) * Dim1);

        }
    };
#endif
}//namespace gridtools
