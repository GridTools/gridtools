#ifndef _GCL_ARRAY_H_
#define _GCL_ARRAY_H_

/**
@file
@briefImplementation of an array class
*/

#include <stddef.h>
#include <algorithm>
#include <boost/type_traits/has_trivial_constructor.hpp>

#include "defs.hpp"
#include "gt_assert.hpp"
#include "host_device.hpp"
#include "generic_metafunctions/accumulate.hpp"

namespace gridtools {

    template <typename T, size_t D, class ENABLE=void>
    class array;

    template <typename T, size_t D>
    class array<T,D, typename boost::enable_if<typename boost::has_trivial_constructor<T>::type>::type> {

        static const uint_t _size = (D>0)?D:1;

        T _array[_size];

    public:
        typedef T value_type;

        GT_FUNCTION
        array() {}

#ifdef CXX11_ENABLED

#ifndef __CUDACC__ // NVCC always returns false in the SFINAE
        // variadic constructor enabled only for arguments of type T
        template<typename ... ElTypes
                 , typename = typename boost::enable_if_c<accumulate(logical_and(), boost::is_same<ElTypes, T>::type::value ...), int >
                 >
        GT_FUNCTION constexpr
        array(ElTypes const& ... types): _array{(T)types ... } {
        }
#else // nvcc only checks the first argument
        // variadic constructor enabled only for arguments of type T
        template<typename First, typename ... ElTypes
                 , typename = typename boost::enable_if_c<boost::is_same<First, T>::type::value , int >
                 >
        GT_FUNCTION
        // constexpr
        array(First const& first_, ElTypes const& ... types): _array{(T)first_, (T)types ... } {
        }
#endif
#else
        //TODO provide a BOOST PP implementation for this
        GT_FUNCTION
        array(T const& i): _array() {
            const_cast<typename boost::remove_const<T>::type*>(_array)[0]=i;
        }
        GT_FUNCTION
        array(T const& i, T const& j): _array() {
            const_cast<typename boost::remove_const<T>::type*>(_array)[0]=i;
            const_cast<typename boost::remove_const<T>::type*>(_array)[1]=j;
        }
        GT_FUNCTION
        array(T const& i, T const& j, T const& k): _array() {
            const_cast<typename boost::remove_const<T>::type*>(_array)[0]=i;
            const_cast<typename boost::remove_const<T>::type*>(_array)[1]=j;
            const_cast<typename boost::remove_const<T>::type*>(_array)[2]=k;
        }
        GT_FUNCTION
        array(T const& i, T const& j, T const& k, T const& l): _array() {
            const_cast<typename boost::remove_const<T>::type*>(_array)[0]=i;
            const_cast<typename boost::remove_const<T>::type*>(_array)[1]=j;
            const_cast<typename boost::remove_const<T>::type*>(_array)[2]=k;
            const_cast<typename boost::remove_const<T>::type*>(_array)[3]=l;
        }
        GT_FUNCTION
        array(T const& i, T const& j, T const& k, T const& l, T const& p): _array() {
            const_cast<typename boost::remove_const<T>::type*>(_array)[0]=i;
            const_cast<typename boost::remove_const<T>::type*>(_array)[1]=j;
            const_cast<typename boost::remove_const<T>::type*>(_array)[2]=k;
            const_cast<typename boost::remove_const<T>::type*>(_array)[3]=l;
            const_cast<typename boost::remove_const<T>::type*>(_array)[4]=p;
        }

#endif

#ifdef CXX11_ENABLED
        /** @brief constexpr copy constructor

        */
        GT_FUNCTION
        constexpr array( array<T,1> const& other): _array{other[0]} {
        }
        GT_FUNCTION
        constexpr array( array<T,2> const& other): _array{other[0], other[1]} {
        }
        GT_FUNCTION
        constexpr array( array<T,3> const& other): _array{other[0], other[1], other[2]}{
        }
        GT_FUNCTION
        constexpr array( array<T,4> const& other): _array{other[0], other[1], other[2], other[3]} {
        }
#else
        //TODO provide a BOOST PP implementation for this (so ugly :-()
        GT_FUNCTION
        array( array<T,1> const& other): _array() {
            _array[0]=other[0];
        }
        GT_FUNCTION
        array( array<T,2> const& other): _array() {
            _array[0]=other[0];
            _array[1]=other[1];
        }
        GT_FUNCTION
        array( array<T,3> const& other): _array() {
            _array[0]=other[0];
            _array[1]=other[1];
            _array[2]=other[2];
        }
        GT_FUNCTION
        array( array<T,4> const& other): _array() {
            _array[0]=other[0];
            _array[1]=other[1];
            _array[2]=other[2];
            _array[3]=other[3];
        }
        GT_FUNCTION
        array( array<T,5> const& other): _array() {
            _array[0]=other[0];
            _array[1]=other[1];
            _array[2]=other[2];
            _array[3]=other[3];
            _array[4]=other[4];
        }
#endif

        GT_FUNCTION
        T * data() const {
            return _array;
        }

        GT_FUNCTION
        constexpr T const & operator[](size_t i) const {
            // assert((i < _size));
            return _array[i];
        }

        GT_FUNCTION
        // constexpr
        T & operator[](size_t i) {
            //assert((i < _size));
            return _array[i];
        }

        template <typename A>
        GT_FUNCTION
        array& operator=(A const& a) {
            assert(a.size() == _size);
            std::copy(a.begin(), a.end(), _array);
            return this;
        }

        GT_FUNCTION
        static constexpr size_t size() {return _size;}
    };

    template <typename T, size_t D>
    class array<T,D, typename boost::disable_if<typename boost::has_trivial_constructor<T>::type>::type > {

        static const uint_t _size = (D>0)?D:1;

        struct _data_item {
            char _data_storage[sizeof(T)];
        };

        _data_item _array[_size];

    public:
        typedef T value_type;

        GT_FUNCTION
        array() {}

#ifdef CXX11_ENABLED
        array(std::initializer_list<T> c) {
            assert(c.size() == _size);
            std::copy(c.begin(), c.end(), _array);
        }
#endif

        GT_FUNCTION
        T const & operator[](size_t i) const {
            assert((i < _size));
            return *(reinterpret_cast<const T*>(&(_array[i])));
        }

        GT_FUNCTION
        T & operator[](size_t i) {
            assert((i < _size));
            return *(reinterpret_cast<T*>(&(_array[i])));
        }

        template <typename A>
        GT_FUNCTION
        array& operator=(A const& a) {
            assert(a.size() == _size);
            std::copy(a.begin(), a.end(), _array);
            return *this;
        }

        GT_FUNCTION
        static constexpr size_t size() {return _size;}
    };

    template<typename T> struct is_array : boost::mpl::false_{};

    template <typename T, size_t D, class ENABLE>
    struct is_array <array<T, D, ENABLE> > : boost::mpl::true_{};

} // namespace gridtools

#endif
