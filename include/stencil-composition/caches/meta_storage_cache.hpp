#pragma once
#include "../../storage/meta_storage_base.hpp"
#include "../../common/generic_metafunctions/unzip.hpp"

namespace gridtools{
    template <typename Layout, uint_t ... Dims>
    struct meta_storage_cache{

        typedef meta_storage_base<0, Layout, false> meta_storage_t;

#ifdef __CUDACC__
        GT_FUNCTION
        static constexpr meta_storage_base<0, Layout, false> value(){return meta_storage_t{Dims ...};}

        GT_FUNCTION
        static constexpr uint_t const& size() {return meta_storage_t{Dims ...}.size();}

        template <typename Accessor>
        GT_FUNCTION
        static constexpr int_t index(Accessor const& arg_) {return meta_storage_t{Dims ...}._index(arg_);}
#else
        static constexpr meta_storage_t m_value{Dims...};

        GT_FUNCTION
        static constexpr meta_storage_base<0, Layout, false> value(){return m_value;}

        GT_FUNCTION
        static constexpr uint_t const& size() {return m_value.size();}

        template <typename Accessor>
        GT_FUNCTION
        static constexpr int_t index(Accessor const& arg_) {return m_value._index(arg_);}
#endif

    private:
        constexpr meta_storage_cache(){};
    };

#ifndef __CUDACC__
    template <typename Layout, uint_t ... Dims>
    constexpr typename meta_storage_cache<Layout, Dims...>::meta_storage_t meta_storage_cache<Layout, Dims...>::m_value;
#endif
}//namespace gridtools
