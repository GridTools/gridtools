#pragma once

template <int I, typename LocationType>
struct arg {
    using location_type = LocationType;
};

template <int I, typename T>
std::ostream& operator<<(std::ostream& s, arg<I,T>) {
    return s << "placeholder<" << I << ", " << T() << ">";
}

