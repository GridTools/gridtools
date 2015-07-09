/*
 * File:   test_domain.cpp
 * Author: mbianco
 *
 * Created on February 5, 2014, 4:16 PM
 *
 * Test domain features, especially the working on the GPU
 */

#define BOOST_NO_CXX11_RVALUE_REFERENCES

#include <boost/type_traits/is_integral.hpp>
#include <common/meta_array.hpp>
#include <boost/mpl/vector/vector10.hpp>
/*
 *
 */
bool test_meta_array_elements() {

//    typedef meta_array<boost::mpl::vector4<int, int, int, int> > metaArray;
//
//    return boost::is_same<typename meta_array_elements_type<metaArray>::type, int>::value;
}
template<typename T> struct is_int : boost::mpl::false_{};
template<> struct is_int<int> : boost::mpl::true_{};

bool test_is_meta_array_of() {

    typedef gridtools::meta_array<boost::mpl::vector4<int, int, int, long>, boost::mpl::quote1<boost::is_integral > > metaArray2;
//
//    typedef meta_array<boost::mpl::vector4<int, int, int, int> > metaArray;
//    return is_meta_array_of<metaArray, boost::mpl::quote1<is_int > >::value;
}
