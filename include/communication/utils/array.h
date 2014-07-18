#ifndef _GCL_ARRAY_H_
#define _GCL_ARRAY_H_

#include <stddef.h>
#include <assert.h>
#include <algorithm>
#include <boost/type_traits/has_trivial_constructor.hpp>
#include <boost/utility/enable_if.hpp>
namespace GCL {

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
      //std::copy(a.begin(), a.end(), _array);
      std::copy(_array, _array+_size, a.begin());
      return *this;
    }

    // copy constructor
    __host__ __device__
    array(array const& a) {
      assert(a.size() == _size);
      int i=0;
      const T* p=a.begin();
      for( ; p < a.end(); ++p) {
          _array[i++] = *p;
      }
      //std::copy(a.begin(), a.end(), _array);
    }

    __host__ __device__
    T* begin(){
        return &_array[0];
    }

    __host__ __device__
    T* end(){
        return &_array[_size];
    }

    __host__ __device__
    const T * begin() const {
        return &_array[0];
    }

    __host__ __device__
    const T * end() const {
        return &_array[_size];
    }

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

    // copy constructor
    /*
    array(array const& a) {
      assert(a.size() == _size);
      std::copy(a.begin(), a.end(), _array);
    }

    __host__ __device__
    T* begin(){
        return &_array[0];
    }

    __host__ __device__
    T* end(){
        return &_array[_size];
    }
    */

    size_t size() const {return _size;}
  };
}

#endif
