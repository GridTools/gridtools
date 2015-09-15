#pragma once
#include "common/defs.hpp"
#include "stencil-composition/arg_metafunctions_fwd.hpp"

namespace gridtools {

template <uint_t I, typename LocationType>
struct arg {
    using location_type = LocationType;

//    template<typename Storage>
//    gridtools::arg_storage_pair<arg<I,LocationType>, Storage>
//    operator=(Storage& ref) {

//        return gridtools::arg_storage_pair<arg<I,LocationType>, Storage>(&ref);
//    }
};

template<uint_t I, typename LocationType>
struct is_arg<arg<I, LocationType> > : boost::mpl::true_{};

template <int I, typename T>
std::ostream& operator<<(std::ostream& s, arg<I,T>) {
    return s << "placeholder<" << I << ", " << T() << ">";
}

}
