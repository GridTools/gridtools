#pragma once

namespace gridtools{

    /**@brief Class in substitution of std::pow, not available in CUDA*/
    template <uint_t Number>
    struct pow{
        template<typename Value>
        GT_FUNCTION
        static Value constexpr apply(Value const& v)
            {
                return v*pow<Number-1>::apply(v);
            }
    };

    /**@brief Class in substitution of std::pow, not available in CUDA*/
    template <>
    struct pow<0>{
        template<typename Value>
        GT_FUNCTION
        static Value constexpr apply(Value const& v)
            {
                return 1.;
            }
    };
} //namespace gridtools

