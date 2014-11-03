#pragma once
#include<common/defs.h>
#include"base_storage.h"

namespace gridtools {
    template < typename BaseStorage >
      struct storage : public /*BaseStorage, */clonable_to_gpu<storage<BaseStorage> >
    {
      typedef typename BaseStorage::basic_type basic_type;
      typedef storage<BaseStorage> original_storage;
      typedef clonable_to_gpu<storage<BaseStorage> > gpu_clone;
      typedef typename BaseStorage::iterator_type iterator_type;
      typedef typename BaseStorage::value_type value_type;
      static const ushort_t n_args = basic_type::n_args;

      __device__
	storage(storage const& other)
	  :  m_base_storage(other.m_base_storage)
      {}
      
      // ~storage(){}
      // storage():BaseStorage(){}

    explicit storage(uint_t dim1, uint_t dim2, uint_t dim3,
		     typename BaseStorage::value_type init = BaseStorage::value_type(), std::string const& s = std::string("default name") ): m_base_storage(dim1, dim2, dim3, init, s) {
        }

      GT_FUNCTION
      void copy_data_to_gpu(){m_base_storage.copy_data_to_gpu();}
      void h2d_update(){m_base_storage.h2d_update();}
      void d2h_update(){m_base_storage.d2h_update();}
      GT_FUNCTION
      typename BaseStorage::const_iterator_type min_addr() const {return m_base_storage.min_addr();}
      GT_FUNCTION
      typename BaseStorage::const_iterator_type max_addr() const { return m_base_storage.max_addr();}
      GT_FUNCTION
      typename BaseStorage::value_type& operator()(uint_t i, uint_t j, uint_t k) { return m_base_storage(i,j,k); }
      GT_FUNCTION
      typename BaseStorage::value_type const & operator()(uint_t i, uint_t j, uint_t k) const {return m_base_storage(i,j,k); }
      GT_FUNCTION
      int_t offset(int_t i, int_t j, int_t k) const {return m_base_storage.offset(i,j,k);}
      GT_FUNCTION
      inline uint_t size() const {return m_base_storage.size();}
      void print_value(uint_t i, uint_t j, uint_t k){ m_base_storage.print_value(i,j,k);}
      GT_FUNCTION
      uint_t _index(uint_t i, uint_t j, uint_t k) const { return m_base_storage._index(i,j,k);}
      template <uint_t Coordinate>
      GT_FUNCTION
      void increment(uint_t* index){m_base_storage.template increment<Coordinate>(index);}
      template <uint_t Coordinate>
      GT_FUNCTION
      void inline increment(uint_t const& dimension, uint_t* index){ m_base_storage.template increment<Coordinate>(dimension, index);}
      GT_FUNCTION
      typename BaseStorage::pointer_type data() const {return m_base_storage.data();}
      GT_FUNCTION
      typename BaseStorage::pointer_type::pointee_t* get_address() const {return m_base_storage.get_address();}
      GT_FUNCTION
      bool is_set() const {return m_base_storage.is_set;}

      GT_FUNCTION
      uint_t const& dims(short_t i) const {return m_base_storage.m_dims[i];}

      GT_FUNCTION
      uint_t const& strides(short_t i) const {return m_base_storage.m_strides[i];}

      GT_FUNCTION
      ushort_t const& lru() const {return 0;}

      GT_FUNCTION
      typename BaseStorage::pointer_type const* fields() const {return m_base_storage.fields();}

      GT_FUNCTION
      static constexpr ushort_t get_index_address(short_t offset, ushort_t lru) {
	return (lru+offset+2)%2;
      }

      // GT_FUNCTION
      // std::string const& name() const { return m_base_storage.name(); }


      BaseStorage m_base_storage;

    private : 
      storage();
    };

}//namespace gridtools
