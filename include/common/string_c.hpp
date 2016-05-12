#pragma once

namespace gridtools {

#ifdef CXX11_ENABLED
    /**@file
    @brief implementation of a compile time string.

    It consists of a sequence of static const char* and of a "callable" operator, which can be i.e. calling printf, or
    doing
    compile-time operations.
    */

    template < char const *S >
    struct static_string {
        static const constexpr char *value = S;
    };

    /**@brief this struct allows to perform operations on static char arrays (e.g. print them as concatenated strings)
     */
    template < typename Callable, typename... Known >
    struct string {

        GT_FUNCTION
        constexpr string() {}

        // operator calls the constructor of the arg_type
        GT_FUNCTION
        static void apply() { Callable::apply(Known::value...); }
    };

    /**@brief this struct allows to perform operations on static char arrays (e.g. print them as concatenated strings)
     */
    template < typename Callable, char const *... Known >
    struct string_c {

        GT_FUNCTION
        constexpr string_c() {}

        // static constexpr char* m_known[]={Known...};
        // operator calls the constructor of the arg_type
        GT_FUNCTION
        static void apply() { Callable::apply(Known...); }
    };

    /** apply the given operator to all strings recursively*/
    template < typename String1, typename String2 >
    struct concatenate {

        GT_FUNCTION
        static void apply() {
            String1::to_string::apply();
            String2::to_string::apply();
        }
    };

    /**@brief struct to recursively print all the strings contained in the gridtools::string template arguments*/
    struct print {

        static void apply() {}

        template < typename... S >
        GT_FUNCTION static void apply(const char *first, S... s) {
            printf("%s", first);
            apply(s...);
        }

        template < typename IntType,
            typename... S,
            typename boost::enable_if< typename boost::is_integral< IntType >::type, int >::type = 0 >
        GT_FUNCTION static void apply(IntType first, S... s) {
            printf("%d", first);
            apply(s...);
        }

        template < typename FloatType,
            typename... S,
            typename boost::enable_if< typename boost::is_floating_point< FloatType >::type, int >::type = 0 >
        GT_FUNCTION static void apply(FloatType first, S... s) {
            printf("%f", first);
            apply(s...);
        }
    };
#endif // CXX11_ENABLED

    /**@brief simple function that copies a string **/
    inline char const *malloc_and_copy(char const *src) {
        char *dst = new char[strlen(src) + 1];
        strcpy(dst, src);
        return dst;
    }
} // namespace gridtools
