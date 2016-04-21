#define BOOST_NO_CXX11_RVALUE_REFERENCES

#include "gtest/gtest.h"

#include "common/generic_metafunctions/variadic_typedef.hpp"

using namespace gridtools;

TEST(variadic_typedef, test) {

    typedef variadic_typedef<int,double, unsigned int> tt;

    GRIDTOOLS_STATIC_ASSERT((boost::is_same<tt::template get_elem<0>::type, int>::value), "Error");

    GRIDTOOLS_STATIC_ASSERT((boost::is_same<tt::template get_elem<1>::type, double>::value), "Error");

    GRIDTOOLS_STATIC_ASSERT((boost::is_same<tt::template get_elem<2>::type, unsigned int>::value), "Error");

}
