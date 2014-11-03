#pragma once
#include <boost/lexical_cast.hpp>

#include "../common/basic_utils.h"
#include "../common/gpu_clone.h"
#include "../common/gt_assert.h"
#include "../common/is_temporary_storage.h"
#include "../stencil-composition/backend_traits_host.h"
#include "../stencil-composition/backend_traits_cuda.h"
#include "hybrid_pointer.h"
#include <iostream>

/**@file
   @brief Implementation of the main storage class, used by all backends, for temporary and non-temporary storage
 */
namespace gridtools {



    namespace _impl
    {
        template <ushort_t I, typename OtherLayout, int_t X>
        struct get_stride_
        {
            GT_FUNCTION
            static uint_t get(const uint_t* s) {
                return s[OtherLayout::template at_<I>::value];
            }
        };

        template <ushort_t I, typename OtherLayout>
        struct get_stride_<I, OtherLayout, 2>
        {
            GT_FUNCTION
            static uint_t get(const uint_t* ) {
#ifndef __CUDACC__
#ifndef NDEBUG
                //                std::cout << "U" ;//<< std::endl;
#endif
#endif
                return 1;
            }
        };

        template <ushort_t I, typename OtherLayout>
        struct get_stride
          : get_stride_<I, OtherLayout, OtherLayout::template at_<I>::value>
        {};

/**@brief Functor updating the pointers on the device */
        struct update_pointer {

#ifdef __CUDACC__
	  template < typename StorageType//typename T, typename U, bool B
                      >
            // GT_FUNCTION_WARNING
	  __host__
	  void operator()(/*base_storage<enumtype::Cuda,T,U,B
                            >*/StorageType *& s) const {
                if (s) {
		  s->copy_data_to_gpu();
		  s->clone_to_gpu();
		  s = s->gpu_object_ptr;
                }
            }
#else
            template <typename StorageType>
            GT_FUNCTION_WARNING
            void operator()(StorageType* s) const {}
#endif

        };
    }//namespace _impl

    namespace _debug{
#ifndef NDEBUG
        struct print_pointer {
            template <typename StorageType>
            GT_FUNCTION_WARNING
            void operator()(StorageType* s) const {
                printf("CIAOOO TATATA %x\n",  s);
            }

#ifdef __CUDACC__
            template < typename T, typename U, bool B
                      >
            GT_FUNCTION_WARNING
            void operator()(base_storage<enumtype::Cuda,T,U,B
                            > *& s) const {
                printf("CIAO POINTER %X\n", s);
            }
#endif
        };
#endif
    }//namespace _debug


/**
   @biref main class for the storage
 */
    template < enumtype::backend Backend,
               typename ValueType,
               typename Layout,
               bool IsTemporary = false
               >
    struct base_storage// : public clonable_to_gpu<base_storage<Backend, ValueType, Layout, IsTemporary> > 
    {
        typedef Layout layout;
        typedef ValueType value_type;
        typedef value_type* iterator_type;
        typedef value_type const* const_iterator_type;
        typedef backend_from_id <Backend> backend_traits_t;
        typedef typename backend_traits_t::template pointer<value_type>::type pointer_type;
	//typedef in order to stopo the type recursion of the derived classes
	typedef base_storage<Backend, ValueType, Layout, IsTemporary> basic_type;
	typedef base_storage<Backend, ValueType, Layout, IsTemporary> original_storage;
        static const ushort_t n_args = 1;

    public:
        explicit base_storage(uint_t dim1, uint_t dim2, uint_t dim3,
                              value_type init = value_type(), std::string const& s = std::string("default name") ):
            m_size( dim1 * dim2 * dim3 ),
            is_set( true ),
            m_data( m_size )//,
	// m_name(s)
            {
            m_dims[0]=( dim1 );
            m_dims[1]=( dim2 );
            m_dims[2]=( dim3 );
	    // printf("layout: %d %d %d \n", layout::get(0), layout::get(1), layout::get(2));
            m_strides[0]=( m_dims[layout::get(2)]*m_dims[layout::get(1)]);
            m_strides[1]=( m_dims[layout::get(2)] );
            m_strides[2]=( 1 );
#ifdef _GT_RANDOM_INPUT
            srand(12345);
#endif
            //m_data = new value_type[m_size];
            for (uint_t i = 0; i < m_size; ++i)
#ifdef _GT_RANDOM_INPUT
                    m_data[i] = init * rand();
#else
                m_data[i] = init;
#endif
                m_data.update_gpu();
            }

      template<typename T>
      __device__
      base_storage(T const& other)
      	  : m_size(other.size())
      	  , is_set(other.is_set)
      	  // , m_name(other.name())
      	  , m_data(other.data())
        {
      	  m_dims[0] = other.dims(0);
      	  m_dims[1] = other.dims(1);
      	  m_dims[2] = other.dims(2);

      	  m_strides[0] = other.strides(0);
      	  m_strides[1] = other.strides(1);
      	  m_strides[2] = other.strides(2);
        }


        __device__
        base_storage(base_storage const& other)
            : m_size(other.m_size)
            , is_set(other.is_set)
            // , m_name(other.m_name)
            , m_data(other.m_data)
        {
            m_dims[0] = other.m_dims[0];
            m_dims[1] = other.m_dims[1];
            m_dims[2] = other.m_dims[2];

            m_strides[0] = other.m_strides[0];
            m_strides[1] = other.m_strides[1];
            m_strides[2] = other.m_strides[2];
        }

      GT_FUNCTION
      void copy_data_to_gpu() const {data().update_gpu();}

      explicit base_storage(): /*m_name("default_name"), */m_data((value_type*)NULL){
            is_set=false;
        }


        // std::string const& name() const {
        //     return m_name;
        // }

        static void text() {
            std::cout << BOOST_CURRENT_FUNCTION << std::endl;
        }

        void h2d_update(){
            m_data.update_gpu();
            }

        void d2h_update(){
            m_data.update_cpu();
        }

        void info() const {
            // std::cout << m_dims[0] << "x"
            //           << m_dims[1] << "x"
            //           << m_dims[2] << ", "
            //           << std::endl;
        }

        GT_FUNCTION
        const_iterator_type min_addr() const {
            return &(m_data[0]);
        }


        GT_FUNCTION
        const_iterator_type max_addr() const {
            return &(m_data[m_size]);
        }

        GT_FUNCTION
        value_type& operator()(uint_t i, uint_t j, uint_t k) {
            /* std::cout<<"indices= "<<i<<" "<<j<<" "<<k<<std::endl; */
	  backend_traits_t::assertion(_index(i,j,k) >= 0);
	  backend_traits_t::assertion(_index(i,j,k) < m_size);
	  return m_data[_index(i,j,k)];
        }


        GT_FUNCTION
        value_type const & operator()(uint_t i, uint_t j, uint_t k) const {
            backend_traits_t::assertion(_index(i,j,k) >= 0);
            backend_traits_t::assertion(_index(i,j,k) < m_size);
            return m_data[_index(i,j,k)];
        }

        template <ushort_t I>
        GT_FUNCTION
        uint_t stride_along() const {
            return _impl::get_stride<I, layout>::get(m_strides); /*layout::template at_<I>::value];*/
        }

        //note: offset returns a signed int because the layout map indexes are signed short ints
        GT_FUNCTION
        int_t offset(int_t i, int_t j, int_t k) const {
	  return //layout::template find<2>(m_dims) * layout::template find<1>(m_dims)
            m_strides[0]* layout::template find<0>(i,j,k) +
	  //layout::template find<2>(m_dims)
	  m_strides[1] * layout::template find<1>(i,j,k) +
            layout::template find<2>(i,j,k);
        }

	GT_FUNCTION
	inline uint_t size() const {
	  return m_size;
	}


	// GT_FUNCTION
	// inline uint_t const* strides() const {return m_strides;}


	GT_FUNCTION
	inline uint_t stride_k() const {
	  return layout::template find<2>(m_strides);//e.g. (GPU test) =512*512=262144
	}

        void print() const {
            print(std::cout);
        }
	void print_value(uint_t i, uint_t j, uint_t k){ printf("value(%d, %d, %d)=%f, at index %d on the data\n", i, j, k, m_data[_index(i, j, k)], _index(i, j, k));}

    static const std::string info_string;

//private:
        uint_t m_dims[3];
        uint_t m_strides[3];
        uint_t m_size;
        bool is_set;
        // const std::string& m_name;

        template <typename Stream>
        void print(Stream & stream) const {
            //std::cout << "Printing " << name << std::endl;
            stream << "(" << m_dims[0] << "x"
                      << m_dims[1] << "x"
                      << m_dims[2] << ")"
                      << std::endl;
            stream << "| j" << std::endl;
            stream << "| j" << std::endl;
            stream << "v j" << std::endl;
            stream << "---> k" << std::endl;

            ushort_t MI=12;
            ushort_t MJ=12;
            ushort_t MK=12;

            for (uint_t i = 0; i < m_dims[0]; i += std::max(( uint_t)(1),m_dims[0]/MI)) {
                for (uint_t j = 0; j < m_dims[1]; j += std::max(( uint_t)1,m_dims[1]/MJ)) {
                    for (uint_t k = 0; k < m_dims[2]; k += std::max(( uint_t)1,m_dims[2]/MK)) {
                        stream << "["/*("
                                          << i << ","
                                          << j << ","
                                          << k << ")"*/
                                  << this->operator()(i,j,k) << "] ";
                    }
                    stream << std::endl;
                }
                stream << std::endl;
            }
            stream << std::endl;
        }

        GT_FUNCTION
        uint_t _index(uint_t i, uint_t j, uint_t k) const {
            uint_t index;
            if (IsTemporary) {
                index =
		  //layout::template find<2>(m_dims) * layout::template find<1>(m_dims)
                    m_strides[0]* (modulus(layout::template find<0>(i,j,k),layout::template find<0>(m_dims))) +
		  //layout::template find<2>(m_dims)
		  m_strides[1]*modulus(layout::template find<1>(i,j,k),layout::template find<1>(m_dims)) +
                    modulus(layout::template find<2>(i,j,k),layout::template find<2>(m_dims));
            } else {
                index =
		//layout::template find<2>(m_dims) * layout::template find<1>(m_dims)
		m_strides[0]
                    * layout::template find<0>(i,j,k) +
		//layout::template find<2>(m_dims)
		m_strides[1]* layout::template find<1>(i,j,k) +
                    layout::template find<2>(i,j,k);
		  // m_strides[0]*layout::template find<0>(i,j,k)+m_strides[1]*layout::template find<1>(i,j,k)+m_strides[2]*layout::template find<2>(i,j,k);
            }
            //assert(index >= 0);
            // assert(index <m_size);
            return index;
        }

      template <uint_t Coordinate>
        GT_FUNCTION
      void increment(uint_t* index){
	/* printf("index before k increment = %d\n", *m_index); */
	*index+=layout::template find<Coordinate>(m_strides);
	/* printf("coordinate = %d, index  = %d \n", Coordinate, index);  */
      }

      template <uint_t Coordinate>
        GT_FUNCTION
	void decrement(uint_t const& coordinate, uint_t* index){
	*index-=layout::template find<Coordinate>(m_strides);
      }

      template <uint_t Coordinate>
        GT_FUNCTION
	void inline increment(uint_t const& dimension, uint_t* index){
	//printf("index before = %d", *m_index);
	*index+=layout::template find<Coordinate>(m_strides)*dimension;
	/* printf("dimension = %d, coordinate = %d, index  = %d , stride %d \n", dimension, Coordinate, *index, layout::template find<Coordinate>(m_strides));   */
      }

      template <uint_t Coordinate>
        GT_FUNCTION
	void decrement(uint_t dimension, uint_t* index){
	*index-=layout::template find<Coordinate>(m_strides)*dimension;
      }

        GT_FUNCTION
        pointer_type const& data() const {return m_data;}

        GT_FUNCTION
        typename pointer_type::pointee_t* get_address() const {
            return m_data.get();}

        GT_FUNCTION
	pointer_type const* fields() const {return &m_data;}

      GT_FUNCTION
      uint_t dims(ushort_t i) const {return m_dims[i];}

      GT_FUNCTION
      uint_t strides(ushort_t i) const {return m_strides[i];}

      pointer_type m_data;

    protected:

    };


    struct print_index{
      template <typename BaseStorage>
      GT_FUNCTION
      void operator()(BaseStorage* b ) const {printf("index -> %d, address %lld, 0x%08x \n", b->index(), &b->index(), &b->index());}
    };

    // struct set_index{
    //     GT_FUNCTION
    // 	set_index(uint_t& index):m_index(index){  }

    //   template <typename BaseStorage>
    //   GT_FUNCTION
    //   void operator()(BaseStorage* b ) const {b->set_index(m_index);}
    // private:
    //   uint_t& m_index;
    // };

  template <uint_t Coordinate>
  struct incr{

    GT_FUNCTION
    incr(){};

    template<typename BaseStorage>
    GT_FUNCTION
    void operator()(BaseStorage * b) const {b->template increment<Coordinate>();}
  };

  template <uint_t Coordinate>
  struct incr_stateful{

    GT_FUNCTION
    incr_stateful(uint dimension):m_dimension(dimension){};

    template<typename BaseStorage>
    GT_FUNCTION
    void operator()(BaseStorage * b) const {b->template increment<Coordinate>(m_dimension);}
  private:
    uint_t m_dimension;
  };

  template <uint_t Coordinate>
  struct decr{
    template<typename BaseStorage>
    GT_FUNCTION
    void operator()(BaseStorage * b) const {b->template decrement<Coordinate>();}
  };


/** @brief finite difference discretization struct
    the goal of this struct is to gain ownership of the storage of the previous solutions,
    and to implement a cash for the solutions, where the new one is stored in place of the
    least recently used one (m_lru).*/

    template < typename Storage, ushort_t ExtraDimensions>
    struct extend : public Storage,  clonable_to_gpu<extend<Storage, ExtraDimensions> > //  : public Storage
    {

      typedef Storage super;
      typedef typename super::pointer_type pointer_type;

      typedef typename super::original_storage original_storage;
        typedef typename super::iterator_type iterator_type;
        typedef typename super::value_type value_type;
        explicit extend(uint_t dim1, uint_t dim2, uint_t dim3,
                            value_type init = value_type(), std::string const& s = std::string("default name") ): super(dim1, dim2, dim3, init, s), m_lru(0), m_fields(/*{(value_type*)NULL}*/) {
            push_back(super::m_data);//first solution is the initialization by default
        }

      __device__
      extend(extend const& other)
	: super(other),
	  m_lru(0){
	for (uint_t i=0; i<n_args; ++i)
	  m_fields[i]=super::pointer_type(other.m_fields[i]);
      }

      __host__
      void copy_data_to_gpu(){
	//the fields are otherwise not copied to the gpu, since they are not inserted in the storage_pointers fusion vector
	for (uint_t i=0; i<n_args; ++i)
	  m_fields[i].update_gpu();
      }

        // extend() : m_lru(0), m_fields(){
        // };

        GT_FUNCTION
        typename pointer_type::pointee_t* get_address() const {
            return super::get_address();}

	
	/**note that I pass in the lru integer in order to be able to define it as a constexpr static method*/
        GT_FUNCTION
	  static constexpr ushort_t get_index_address (short_t const& offset, ushort_t const& lru) {
	  return (lru+offset+n_args)%n_args;
	}

        //template <ushort_t Offset>
        GT_FUNCTION
        typename pointer_type::pointee_t* get_address(short_t offset) const {
            //printf("the offset: %d\n", offset);
	  // printf("storage used: %d\n", (m_lru+offset+n_args)%n_args);
	  // printf("GPU address of the data", m_fields[(m_lru+offset+n_args)%n_args].get());
	  return m_fields[(m_lru+offset+n_args)%n_args].get();}
        //super::pointer_type const& data(){return m_fields[lru];}
        GT_FUNCTION
        inline  pointer_type const& get_field(int index) const {return std::get<index>(m_fields);};
        //the time integration takes ownership over all the pointers?
        GT_FUNCTION
        inline void swap(/*smart<*/ pointer_type/*>*/ & field){
            //cycle in a ring
            pointer_type swap(m_fields[m_lru]);
            m_fields[m_lru]=field;
            field = swap;
            m_lru=(m_lru+1)%n_args;
            this->m_data=m_fields[m_lru];
        }

        GT_FUNCTION
        inline void push_back(/*smart<*/ pointer_type/*>*/ & field){
            //cycle in a ring
            m_fields[m_lru]=field;
            m_lru=(m_lru+1)%n_args;
            this->m_data=m_fields[m_lru];
        }

        //the time integration takes ownership over all the pointers?
        GT_FUNCTION
        inline void advance(short_t offset=1){
            //cycle in a ring
            m_lru=(m_lru+n_args+offset)%n_args;
            this->m_data=m_fields[m_lru];
        }

      GT_FUNCTION
	inline pointer_type const*  fields(){return m_fields;}

      GT_FUNCTION
	inline ushort_t const& lru(){return m_lru;}

        void print() {
            print(std::cout);
        }

        template <typename Stream>
        void print(Stream & stream) {
            for(ushort_t i=0; i < n_args; ++i)
            {
                super::print(stream);
                advance();
            }
        }

        static const ushort_t n_args = ExtraDimensions+1;
    private:
        ushort_t m_lru;
        //todo: create a smart<Pointer> adding the policy to whatever pointer.
        /*smart<*/pointer_type/*>*/ m_fields[n_args];
    };


    template < enumtype::backend B, typename ValueType, typename Layout, bool IsTemporary
        >
    const std::string base_storage<B , ValueType, Layout, IsTemporary
            >::info_string=boost::lexical_cast<std::string>("-1");

    template <enumtype::backend B, typename ValueType, typename Y>
    struct is_temporary_storage<base_storage<B,ValueType,Y,false>*& >
      : boost::false_type
    {};

    template <enumtype::backend X, typename ValueType, typename Y>
    struct is_temporary_storage<base_storage<X,ValueType,Y,true>*& >
      : boost::true_type
    {};

    template <enumtype::backend X, typename ValueType, typename Y>
    struct is_temporary_storage<base_storage<X,ValueType,Y,false>* >
      : boost::false_type
    {};

    template <enumtype::backend X, typename ValueType, typename Y>
    struct is_temporary_storage<base_storage<X,ValueType,Y,true>* >
      : boost::true_type
    {};

    template <enumtype::backend X, typename ValueType, typename Y>
    struct is_temporary_storage<base_storage<X,ValueType,Y,false> >
      : boost::false_type
    {};

    template <enumtype::backend X, typename ValueType, typename Y>
    struct is_temporary_storage<base_storage<X,ValueType,Y,true> >
      : boost::true_type
    {};

    //Decorator is the extend
  template <template <typename T, ushort_t ... O> class Decorator, typename BaseType, ushort_t ... Extra>
  struct is_temporary_storage<Decorator<BaseType, Extra...> > : is_temporary_storage< typename BaseType::basic_type >
    {};

    //Decorator is the extend
  template <template <typename T, ushort_t ... O> class Decorator, typename BaseType, ushort_t ... Extra>
  struct is_temporary_storage<Decorator<BaseType, Extra...>* > : is_temporary_storage< typename BaseType::basic_type* >
    {};

    //Decorator is the extend
  template <template <typename T, ushort_t ... O> class Decorator, typename BaseType, ushort_t ... Extra>
  struct is_temporary_storage<Decorator<BaseType, Extra...>& > : is_temporary_storage< typename BaseType::basic_type& >
    {};

    //Decorator is the extend
  template <template <typename T, ushort_t ... O> class Decorator, typename BaseType, ushort_t ... Extra>
  struct is_temporary_storage<Decorator<BaseType, Extra...>*& > : is_temporary_storage< typename BaseType::basic_type*& >
    {};

    template <enumtype::backend Backend, typename T, typename U, bool B>
    std::ostream& operator<<(std::ostream &s, base_storage<Backend,T,U, B> ) {
        return s << "base_storage <T,U," << " " << std::boolalpha << B << "> ";
            }


} //namespace gridtools
