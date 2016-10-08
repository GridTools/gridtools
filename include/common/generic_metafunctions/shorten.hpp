#pragma once
#include "../defs.hpp"

#ifdef CXX11_ENABLED

namespace gridtools {
    namespace impl {
        template < uint_t cnt, typename Value, uint_t Threshold, typename VariadicHolder, Value... Rest >
        struct shorten_impl;

        template < uint_t cnt, typename Value, Value... Args, template < Value... > class VariadicHolder >
        struct shorten_impl< cnt, Value, cnt, VariadicHolder< Args... > > {
            using type = VariadicHolder< Args... >;
        };

        template < uint_t cnt,
            typename Value,
            uint_t Threshold,
            Value... Args,
            Value FirstRest,
            Value... Rest,
            template < Value... > class VariadicHolder >
        struct shorten_impl< cnt, Value, Threshold, VariadicHolder< Args... >, FirstRest, Rest... > {
            using type = typename boost::mpl::eval_if_c< cnt == Threshold,
                boost::mpl::identity< VariadicHolder< Args... > >,
                shorten_impl< cnt + 1, Value, Threshold, VariadicHolder< Args..., FirstRest >, Rest... > >::type;
        };
    }

    /*
     * Given a type with a set of variadic templates, returns the same type with only the
     * first "Threshold" number of variadic templates. Threshold has to be smaller or equal than
     * the number of variadic templates contained in the holder type
     * Example of use:
     *   shorten<int, vector<3,4,5>, 2> == vector<3,4>
     */
    template < typename Value, typename VariadicHolder, uint_t Threshold >
    struct shorten;

    template < typename Value,
        Value First,
        Value... Args,
        template < Value... > class VariadicHolder,
        uint_t Threshold >
    struct shorten< Value, VariadicHolder< First, Args... >, Threshold > {
        static_assert((Threshold <= sizeof...(Args) + 1), "Error");
        using type = typename impl::shorten_impl< 0, Value, Threshold, VariadicHolder<>, First, Args... >::type;
    };
}

#endif
