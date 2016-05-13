#pragma once

namespace gdl{

    namespace gt=gridtools;

    namespace enumtype{
        using namespace gt::enumtype;
        // using gt::enumtype;
        enum Basis {Lagrange, BSplines, Legendre};
        enum Shape {Hexa, Tetra, Quad, Tri,  Line, Point};
    }

    using uint_t = gt::uint_t;
    using int_t = gt::uint_t;
    using short_t = gt::uint_t;
    using ushort_t = gt::uint_t;

    using float_type = gt::float_type;

    template<int_t T>
    using static_int=gt::static_int<T>;
    template<uint_t T>
    using static_uint=gt::static_uint<T>;
    template<short_t T>
    using static_short=gt::static_short<T>;
    template<ushort_t T>
    using static_ushort=gt::static_ushort<T>;


    template<typename T>
    struct zero
    {
        static constexpr T value{T()};
    };

}
