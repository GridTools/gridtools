#ifndef _GCL_ARRAY_H_
#define _GCL_ARRAY_H_

/**
@file
@briefImplementation of an array class
*/

#include <stddef.h>
#include "defs.h"
#include "gt_assert.h"
#include "host_device.h"
#include <algorithm>
#include <boost/type_traits/has_trivial_constructor.hpp>

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
        array(std::initializer_list<T> c) {
            assert(c.size() == _size);
            std::copy(c.begin(), c.end(), _array);
        }
#endif

        GT_FUNCTION
        T * data() const {
            return _array;
        }

            GT_FUNCTION
        T  const & operator[](size_t i) const {
            assert((i>=0) && (i < _size));
            return _array[i];
        }

        GT_FUNCTION
        T & operator[](size_t i) {
            assert((i>=0) && (i < _size));
            return _array[i];
        }

        template <typename A>
        GT_FUNCTION
        array& operator=(A *const& a) {
            _array=a;
            return this;
        }

        template <typename A>
        GT_FUNCTION
        array& operator=(A const& a) {
            assert(a.size() == _size);
            std::copy(a.begin(), a.end(), _array);
            return this;
        }

        GT_FUNCTION
        size_t size() const {return _size;}
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
            assert((i>=0) && (i < _size));
            return *(reinterpret_cast<const T*>(&(_array[i])));
        }

        GT_FUNCTION
        T & operator[](size_t i) {
            assert((i>=0) && (i < _size));
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
        size_t size() const {return _size;}
    };
} // namespace gridtools

#endif
