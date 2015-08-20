#pragma once

#include <stencil-composition/arg_storage_pair.hpp>

template <int I, typename LocationType>
struct arg {
    using location_type = LocationType;

    template<typename Storage>
    gridtools::arg_storage_pair<arg<I,LocationType>, Storage>
    operator=(Storage& ref) {

        return gridtools::arg_storage_pair<arg<I,LocationType>, Storage>(&ref);
    }
};

template <int I, typename T>
std::ostream& operator<<(std::ostream& s, arg<I,T>) {
    return s << "placeholder<" << I << ", " << T() << ">";
}

