#pragma once
#include "common/defs.hpp"
#include "storage/storage_metafunctions.hpp"
#include "stencil-composition/arg_metafunctions_fwd.hpp"

namespace gridtools {

template<typename T> struct is_storage;

template <uint_t I, typename Storage>
struct arg {

    //TODO the is_storage trait is messy
//    GRIDTOOLS_STATIC_ASSERT((is_storage<Storage>::value), "Error: wrong parameter for storage class");
    typedef Storage storage_type;
    typedef typename Storage::iterator_type iterator_type;
    typedef typename Storage::value_type value_type;
    typedef static_uint<I> index_type;
    typedef static_uint<I> index;

    using location_type = typename Storage::meta_data_t::index_type;
};

template<uint_t I, typename Storage>
struct is_arg<arg<I, Storage> > : boost::mpl::true_{};

template <int I, typename T>
std::ostream& operator<<(std::ostream& s, arg<I,T>) {
    return s << "placeholder<" << I << ", " << T() << ">";
}

}
