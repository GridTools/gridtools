#include <iostream>
#include <boost/config.hpp> 
#include <boost/mpl/void.hpp>
#include "hasdo.h"

struct B {
    template <typename T>
    static void Do(T a, char b) {}
    template <typename T>
    static double function(T a, char b) {}
};

namespace letsee {
    struct check {
        static int test(...);
    };

    template<typename X>
    X& operator, (X&, boost::mpl::void_);
    template<typename X>
    const X& operator, (const X&, boost::mpl::void_);

    template <typename F, typename T, typename U>
    struct A {

        struct derived: public F {
            using F::function;
            static int function(...);
        };

        BOOST_STATIC_CONSTANT( bool, value = (sizeof(check::test(derived::function(*(T*)0, *(U*)0)),boost::mpl::void_()) == sizeof(int)) );
    };
} //namespace letsee

int main() {

    std::cout << std::boolalpha << letsee::A<B,int,char>::value << std::endl;

    std::cout << std::boolalpha << has_do<B, char>::value << std::endl;

    return 0;
}


