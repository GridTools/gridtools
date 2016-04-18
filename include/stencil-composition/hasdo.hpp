#pragma once

#include <boost/config.hpp>
#include <boost/mpl/void.hpp>
#include "yesno.hpp"

namespace gridtools {
    /**
     * @struct has_do_member
     * Meta function testing if a functor has a Do member (function, variable or typedef)
     */
    template < typename TFunctor >
    struct has_do_member {
        // define a MixIn class providing a Do member
        struct MixIn {
            void Do() {}
        };
        // multiple inheritance form TFunctor and MixIn
        // (if both define a Do it will be ambiguous in the Derived struct)
        struct Derived : public TFunctor, public MixIn {};

        // define an SFINAE structure
        template < typename TDoFunc, TDoFunc VFunc >
        struct SFINAE {};

        // SFINAE test methods which try to match the MixIn Do method signature
        // (note that multiple inheritance hides all symbols which are defined in more than one base class,
        // i.e. if TFunctor and MixIn both define a Do symbol then it will be ambiguous in the Derived class
        // and we will fall back to the ellipsis test method)
        template < typename TDerived >
        static no test(SFINAE< void (MixIn::*)(), &TDerived::Do > *);
        template < typename TDerived >
        static yes test(...);

        // use the sizeof trick in order to check which overload matches
        BOOST_STATIC_CONSTANT(bool, value = (sizeof(test< Derived >(0)) == sizeof(yes)));
        typedef boost::mpl::bool_< bool(value) > type;
    };

    // setup the HasDoDetails namespace which provides a check function searching for a specific Do method signature
    // (SFINAE is used with a comma operator matching the return value of the Do function in case
    // void is return we fall back to the default comma operator returning the element after the comma)
    namespace HasDoDetails {

        template < bool VCondition, typename TFunctor, typename TInterval >
        struct check_if {
            BOOST_STATIC_CONSTANT(bool, value = false);
            typedef boost::mpl::bool_< bool(value) > type;
        };

        template < typename T >
        yes operator, (T const &, yes);

        no operator, (no const &, yes);

        template < typename F >
        struct embed : F {
            using F::Do;
            static no Do(...);
        };

        // if there is a Do member but no domain parameter check for TResult Do(TArguments)
        template < typename TFunctor, typename TInterval >
        struct check_if< true, TFunctor, TInterval > {

            template < typename Functor, typename Arg, typename Interval >
            struct has_do_impl {
                // this check makes the first argument match anything, const or not
                static const bool value =
                    sizeof((embed< Functor >::Do(*(Arg *)0, *(Interval *)0), yes())) == sizeof(yes);
            };

            static const bool value = has_do_impl< TFunctor, int, TInterval >::value;
        };

        template < bool, typename Functor, typename Interval >
        struct check_if_const {
            static const bool value = sizeof((embed< Functor >::Do(int(), *(Interval *)0), yes())) == sizeof(yes);
        };

        template < typename Functor, typename Interval >
        struct check_if_const< false, Functor, Interval > {
            static const bool value = true;
        };

    } // namespace HasDoDetails

    /**
     * @struct has_do
     * Meta function testing if a functor has a specific Do method
     * (note that the meta function does consider overload resolution as well)
     */
    template < typename TFunctor, typename TInterval >
    struct has_do {
        GRIDTOOLS_STATIC_ASSERT((is_interval< TInterval >::value or is_level< TInterval >::value),
            "has_do second argument must be an interval or a level.");

        BOOST_STATIC_CONSTANT(
            bool, value = (HasDoDetails::check_if< has_do_member< TFunctor >::value, TFunctor, TInterval >::value));
        typedef boost::mpl::bool_< bool(value) > type;

        GRIDTOOLS_STATIC_ASSERT(
            (HasDoDetails::check_if_const< value, TFunctor, TInterval >::value), "Functor signature not compliant");
    };

} // namespace gridtools
