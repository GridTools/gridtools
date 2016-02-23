#pragma once

namespace gridtools{
    /** method replacing the operator ? which selects a branch at compile time and
     allows to return different types whether the condition is true or false */
    template<bool Condition>
    struct static_if;

    template<>
    struct static_if <true>{
        template <typename TrueVal, typename FalseVal>
        GT_FUNCTION
        static constexpr TrueVal& apply(TrueVal& true_val, FalseVal& /*false_val*/)
            {
                return true_val;
            }

        template <typename TrueVal, typename FalseVal>
        GT_FUNCTION
        static constexpr TrueVal const& apply(TrueVal const& true_val, FalseVal const& /*false_val*/)
            {
                return true_val;
            }

        template <typename TrueVal, typename FalseVal>
        GT_FUNCTION
        static void eval(TrueVal const& true_val, FalseVal const& /*false_val*/)
            {
                true_val();
            }

    };

    template<>
    struct static_if <false>{
        template <typename TrueVal, typename FalseVal>
        GT_FUNCTION
        static constexpr FalseVal& apply(TrueVal& /*true_val*/, FalseVal& false_val)
            {
                return false_val;
            }

        template <typename TrueVal, typename FalseVal>
        GT_FUNCTION
        static void eval(TrueVal const& /*true_val*/, FalseVal const& false_val)
            {
                false_val();
            }
};
}
