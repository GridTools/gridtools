#pragma once

namespace gridtools{

    /**@brief Class in substitution of std::gt_pow, not available in CUDA*/
    template <uint_t Number>
    struct gt_pow{
        template<typename Value>
        GT_FUNCTION
        static Value constexpr apply(Value const& v)
            {
                return v*gt_pow<Number-1>::apply(v);
            }
    };

    /**@brief Class in substitution of std::gt_pow, not available in CUDA*/
    template <>
    struct gt_pow<0>{
        template<typename Value>
        GT_FUNCTION
        static Value constexpr apply(Value const& v)
            {
                return 1.;
            }
    };
} //namespace gridtools
