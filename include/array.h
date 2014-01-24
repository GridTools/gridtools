#ifndef _GCL_ARRAY_H_
#define _GCL_ARRAY_H_

#include <stddef.h>
#include <assert.h>
#include <algorithm>
#include <boost/type_traits/has_trivial_constructor.hpp>
#include <boost/utility/enable_if.hpp>
namespace gridtools {

    template <typename T, size_t D, class ENABLE=void>
    class array;

    template <typename T, size_t D>
    class array<T,D, typename boost::enable_if<typename boost::has_trivial_constructor<T>::type>::type > {

        static const int _size = (D>0)?D:1;

        T _array[_size];

    public:
        typedef T value_type;

        __host__ __device__
        array() {}

#ifdef __GXX_EXPERIMENTAL_CXX0X__
        array(std::initializer_list<T> c) {
            assert(c.size() == _size);
            std::copy(c.begin(), c.end(), _array);
        }
#endif

        __host__ __device__
        T const & operator[](size_t i) const {
            return _array[i];
        }

        __host__ __device__
        T & operator[](size_t i) {
            return _array[i];
        }

        template <typename A>
        __host__ __device__
        array& operator=(A const& a) {
            assert(a.size() == _size);
            std::copy(a.begin(), a.end(), _array);
            return this;
        }

        __host__ __device__
        size_t size() const {return _size;}
    };

    template <typename T, size_t D>
    class array<T,D, typename boost::disable_if<typename boost::has_trivial_constructor<T>::type>::type > {

        static const int _size = (D>0)?D:1;

        struct _data_item {
            char _data_storage[sizeof(T)];
        };

        _data_item _array[_size];

    public:
        typedef T value_type;

        __host__ __device__
        array() {}

#ifdef __GXX_EXPERIMENTAL_CXX0X__
        array(std::initializer_list<T> c) {
            assert(c.size() == _size);
            std::copy(c.begin(), c.end(), _array);
        }
#endif

        __host__ __device__
        T const & operator[](size_t i) const {
            return *(reinterpret_cast<const T*>(&(_array[i])));
        }

        __host__ __device__
        T & operator[](size_t i) {
            return *(reinterpret_cast<T*>(&(_array[i])));
        }

        template <typename A>
        __host__ __device__
        array& operator=(A const& a) {
            assert(a.size() == _size);
            std::copy(a.begin(), a.end(), _array);
        }

        __host__ __device__
        size_t size() const {return _size;}
    };
} // namespace gridtools

#endif
