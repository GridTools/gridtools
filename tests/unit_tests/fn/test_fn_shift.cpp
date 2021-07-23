#include <gridtools/fn/shift.hpp>

#include <type_traits>

#include <gtest/gtest.h>

#include <gridtools/common/integral_constant.hpp>

namespace gridtools::fn {
    namespace {
        using namespace literals;

        template <auto...>
        struct dummy {};

        template <auto... Offsets, class Off>
        constexpr dummy<Offsets..., Off{}> fn_shift(dummy<Offsets...>, Off) {
            return {};
        }

        template <auto... Offsets, class Off0, class Off1>
        constexpr dummy<Offsets..., Off0{}, Off1{}> fn_shift(dummy<Offsets...>, Off0, Off1) {
            return {};
        }

        inline constexpr auto zero = shift();
        inline constexpr auto sh = shift(1_c, 2_c, 3_c);

        static_assert(std::is_same_v<decltype(zero(dummy<42_c>{})), dummy<42_c>>);
        static_assert(std::is_same_v<decltype(sh(dummy<42_c>{})), dummy<42_c, 1_c, 2_c, 3_c>>);

        TEST(shift, default) {
            int arr[] = {41, 42, 43};
            EXPECT_EQ(shift(1)(arr), arr + 1);
        }

    } // namespace
} // namespace gridtools::fn
