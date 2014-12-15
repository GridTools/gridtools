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
#include "accumulate.h"

/**@file
   @brief Implementation of the main storage class, used by all backends, for temporary and non-temporary storage

   We define here an important naming convention. We call:

   - the data fields: are contiguous chunks of memory, accessed by 3 (by default, but not necessarily) indexes.
   These structures are univocally defined by 3 (by default) integers. These are currently 2 strides and the total size of the chunks. Note that (in 3D) the relation between these quantities (\f$stride_1\f$, \f$stride_2\f$ and \f$size\f$) and the dimensions x, y and z can be (depending on the storage layout chosen)
   \f[
   size=x*y*z //
   stride_2=x*y //
   stride_1=x .
   \f]
   The quantities \f$size\f$, \f$stride_2\f$ and \f$stride_1\f$ are arranged respectively in m_strides[0], m_strides[1], m_strides[2].
   - the data snapshot: is a single pointer to one data field. The snapshots are arranged in the storages on a 1D array, regardless of the dimension and snapshot they refer to. The arg_type (or arg_decorator) class is
   responsible of computing the correct offests (relative to the given dimension) and address the storages correctly.
   - the storage: is an instance of any storage class, and can contain one or more fields and dimension. Every dimension consists of one or several snaphsots of the fields
   (e.g. if the time T is the current dimension, 3 snapshots can be the fields at t, t+1, t+2)

   The basic_storage class has a 1-1 relation with the data fields, while the subclasses extend the concept of storage to the structure represented in the ASCII picture below. The extension to multiple dimensions
   or to more general cases could be implemented (probably) fairly easily, since the current interface is arbitrary in the number of dimension. The storage layout instead is limited to 2 dimensions, since
   I cannot think of use cases for which more than two dimensions would be necessary.

   NOTE: the constraint of the data fields accessed by the same storage class are the following:
   - the memory layout is one for all the data fields
   - the index used to access the storage position

\verbatim
############### 2D Storage ################
#                    ___________\         #
#                      time     /         #
#                  | |*|*|*|*|*|*|        #
# space, pressure  | |*|*|*|              #
#    energy,...    v |*|*|*|*|*|      	  #
#				          #
#                     ^ ^ ^ ^ ^ ^         #
#                     | | | | | |         #
#                      snapshots          #
#		                 	  #
############### 2D Storage ################
\endverbatim

The final storage which is effectly instantiated must be "clonable to the GPU", i.e. it must derive from the clonable_to_gpu struct.
This is achieved by defining a class with multiple inheritance.

NOTE CUDA: It is important when subclassing from a storage object to reimplement the __device__ copy constructor, and possibly the method 'copy_data_to_gpu' which are used when cloning the class to the CUDA device.
*/

namespace gridtools {


    namespace _impl
    {

/**@brief Functor updating the pointers on the device */
        struct update_pointer {
#ifdef __CUDACC__
            template < typename StorageType//typename T, typename U, bool B
                       >
            GT_FUNCTION_WARNING
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
            template < typename T, typename U, bool B, ushort_t FieldsDimensions
                       >
            GT_FUNCTION_WARNING
            void operator()(base_storage<enumtype::Cuda,T,U,B, FieldsDimensions
                            > *& s) const {
                printf("CIAO POINTER %X\n", s);
            }
#endif
        };
#endif
    }//namespace _debug

    /**@brief metafunction to recursively compute the next stride*/
    template<short_t ID, short_t SpaceDimensions,  typename Layout>
    struct next_stride{
	template<typename First, typename ... IntTypes>
	static First constexpr apply ( First first, IntTypes ... args){
	    return (Layout::template at_<SpaceDimensions-ID+1>::value==vec_max<typename Layout::layout_vector_t>::value)?0:Layout::template find_val<SpaceDimensions-ID,short_t,1>(first, args...) * next_stride<ID-1, SpaceDimensions, Layout>::apply(first, args...);
	}
    };

    /**@brief template specialization to stop the recursion*/
    template< short_t SpaceDimensions, typename Layout>
    struct next_stride<0, SpaceDimensions, Layout>{
	template<typename First, typename ... IntTypes>
	static First constexpr apply(First first, IntTypes ... args){
	    return Layout::template find_val<SpaceDimensions,short_t,1>(first, args...);
	}
    };

    /**@brief metafunction to recursively compute all the strides, in a generic arbitrary dimensional storage*/
    template<int_t ID, int_t SpaceDimensions,  typename Layout>
    struct assign_strides{
	template<typename ... UIntType>
	static void apply(uint_t* strides, UIntType ... args){
	    BOOST_STATIC_ASSERT(SpaceDimensions>=ID);
	    BOOST_STATIC_ASSERT(ID>=0);
	    strides[SpaceDimensions-ID] = next_stride<ID, SpaceDimensions, Layout>::apply(args...);
	    assign_strides<SpaceDimensions-(ID+1), SpaceDimensions, Layout>::apply(strides, args...);
	}
    };

    /**@brief specialization to stop the recursion*/
    template< int_t SpaceDimensions,  typename Layout>
    struct assign_strides<0, SpaceDimensions, Layout>{
	template<typename ... UIntType>
	static void apply(uint_t* strides, UIntType ... args){
	    BOOST_STATIC_ASSERT(SpaceDimensions>=0);
	    strides[SpaceDimensions] = next_stride<0, SpaceDimensions, Layout>::apply(args...);
	}
    };

/**
   @brief main class for the basic storage
   The base_storage class contains one snapshot. It univocally defines the access pattern with three integers: the total storage sizes and the two strides different from one.
*/
    template < enumtype::backend Backend,
               typename ValueType,
               typename Layout,
               bool IsTemporary = false,
	       short_t FieldDimension=1
               >
    struct base_storage
    {
        typedef Layout layout;
        typedef ValueType value_type;
        typedef value_type* iterator_type;
        typedef value_type const* const_iterator_type;
        typedef backend_from_id <Backend> backend_traits_t;
        typedef typename backend_traits_t::template pointer<value_type>::type pointer_type;
        //typedef in order to stopo the type recursion of the derived classes
        typedef base_storage<Backend, ValueType, Layout, IsTemporary, FieldDimension> basic_type;
        typedef base_storage<Backend, ValueType, Layout, IsTemporary, FieldDimension> original_storage;
	static const enumtype::backend backend=Backend;
	static const bool is_temporary = IsTemporary;
	static const ushort_t n_width = 1;
        static const ushort_t space_dimensions = layout::length;
        static const short_t field_dimensions = FieldDimension;

    public:

#ifdef CXX11_ENABLED

	base_storage(uint_t dim1, uint_t dim2, uint_t dim3, value_type init = value_type(), char const* s="default storage"):
	    is_set( true ),
	    m_name(s),
	    m_fields{pointer_type(dim1*dim2*dim3)},
	    m_dims{dim1, dim2, dim3},
	    m_strides{( ((layout::template at_<0>::value < 0)?1:dim1) * ((layout::template at_<1>::value < 0)?1:dim2) * ((layout::template at_<2>::value < 0)?1:dim3) ) ,
		    ( (m_strides[0]<=1)?0:layout::template find_val<2,uint_t,1>(dim1,dim2,dim3)*layout::template find_val<1,short_t,1>(dim1,dim2,dim3) ),
		    ( (m_strides[1]<=1)?0:layout::template find_val<2,uint_t,1>(dim1,dim2,dim3) )}
		    {

#ifdef _GT_RANDOM_INPUT
                srand(12345);
#endif
                for (uint_t i = 0; i < size(); ++i)
#ifdef _GT_RANDOM_INPUT
                    (m_fields[0])[i] = init * rand();
#else
                (m_fields[0])[i] = init;
#endif
    }

#if !defined(__GNUC__) || (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 9) )
	/**@brief generic multidimensional constructor

           The number of arguments must me equal to the space dimensions of the specific field
         */
		    template <class ... UIntTypes>
		    base_storage( UIntTypes const& ... args, value_type init = value_type(), char const* s="default storage" ):
			is_set( true ),
			m_name(s),
			m_dims{args...}
            {
		BOOST_STATIC_ASSERT(sizeof...(UIntTypes)==space_dimensions);
		BOOST_STATIC_ASSERT(field_dimensions>0);
		//static const auto blabla=boost::mpl::print<static_int<field_dimensions> >();
		m_strides[0] = ( accumulate( multiplies(), args...) ),
		    assign_strides<(short_t)(space_dimensions-2), (short_t)(space_dimensions-1), layout>::apply(m_strides, args...);
		m_fields[0]=pointer_type(m_strides[0]);
            }

#endif // GCC<4.9.0

#else //CXX11_ENABLED

	    /**@brief default constructor
	       sets all the data members given the storage dimensions
	    */
	    base_storage(uint_t const& dim1, uint_t const& dim2, uint_t const& dim3,
			 value_type init = value_type(), char const* s="default storage" ):
		is_set( true )
		, m_name(s)
            {
		m_fields[0]=pointer_type(dim1*dim2*dim3);
		m_dims[0]=dim1;
		m_dims[1]=dim2;
		m_dims[2]=dim3;

		m_strides[0]=( ((layout::template at_<0>::value < 0)?1:dim1) * ((layout::template at_<1>::value < 0)?1:dim2) * ((layout::template at_<2>::value < 0)?1:dim3) );
		m_strides[1]=( (m_strides[0]<=1)?0:layout::template find_val<2,short_t,1>(dim1,dim2,dim3)*layout::template find_val<1,short_t,1>(dim1,dim2,dim3) );
		m_strides[2]=( (m_strides[1]<=1)?0:layout::template find_val<2,short_t,1>(dim1,dim2,dim3) );

#ifdef _GT_RANDOM_INPUT
                srand(12345);
#endif
                for (uint_t i = 0; i < size(); ++i)
#ifdef _GT_RANDOM_INPUT
                    (m_fields[0])[i] = init * rand();
#else
                (m_fields[0])[i] = init;
#endif
            }

#endif //CXX11_ENABLED

	/**@brief 3D constructor with the storage pointer provided externally

	 This interface handles the case in which the storage is allocated from the python interface. Since this storege gets freed inside python, it must be instantiated as a
	'managed outside' wrap_pointer. In this way the storage destructor will not free the pointer.*/
        explicit base_storage(uint_t dim1, uint_t dim2, uint_t dim3, value_type* ptr, char const* s="default storage"
	    ):
            is_set( true ),
            m_name(s)
            {
		m_fields[0]=pointer_type(ptr, true);//the storage does not own the pointer
		m_dims[0]=dim1;
		m_dims[1]=dim2;
		m_dims[2]=dim3;
		m_strides[0]=( ((layout::template at_<0>::value < 0)?1:dim1) * ((layout::template at_<1>::value < 0)?1:dim2) * ((layout::template at_<2>::value < 0)?1:dim3) );
		m_strides[1]=( (m_strides[0]==1)?0:layout::template find_val<2,short_t,1>(dim1,dim2,dim3)*layout::template find_val<1,short_t,1>(dim1,dim2,dim3) );
		m_strides[2]=( (m_strides[1]==1)?0:layout::template find_val<2,short_t,1>(dim1,dim2,dim3) );
            }

        /**@brief destructor: frees the pointers to the data fields which are not managed outside */
        virtual ~base_storage(){
	    for(ushort_t i=0; i<field_dimensions; ++i)
		if(!m_fields[i].managed())
		    m_fields[i].free_it();
	}

        /**@brief device copy constructor*/
        template<typename T>
        __device__
        base_storage(T const& other)
            :
	    is_set(other.is_set)
            {
		for (uint_t i=0; i< field_dimensions; ++i)
		    m_fields[i]=pointer_type(other.m_fields[i]);
		m_dims[0]=other.m_dims[0];
		m_dims[1]=other.m_dims[1];
		m_dims[2]=other.m_dims[2];
                m_strides[0] = other.size();
                m_strides[1] = other.strides(1);
                m_strides[2] = other.strides(2);
            }

        /** @brief copies the data field to the GPU */
        GT_FUNCTION_WARNING
        void copy_data_to_gpu() const {
            for (uint_t i=0; i<field_dimensions; ++i)
                m_fields[i].update_gpu();
	}

        static void text() {
            std::cout << BOOST_CURRENT_FUNCTION << std::endl;
        }

        /** @brief update the GPU pointer */
        void h2d_update(){
            for (uint_t i=0; i<field_dimensions; ++i)
                m_fields[i].update_gpu();
        }

        /** @brief updates the CPU pointer */
        void d2h_update(){
            for (uint_t i=0; i<field_dimensions; ++i)
                m_fields[i].update_cpu();
        }

        /** @brief prints debugging information */
        void info() const {
            std::cout << dims<0>() << "x"
                      << dims<1>() << "x"
                      << dims<2>() << ", "
                      << std::endl;
        }

        /** @brief returns the first memory addres of the data field */
        GT_FUNCTION
        const_iterator_type min_addr() const {
            return &((m_fields[0])[0]);
        }


        /** @brief returns the last memry address of the data field */
        GT_FUNCTION
        const_iterator_type max_addr() const {
            return &((m_fields[0])[/*m_size*/m_strides[0]]);
        }

        /** @brief returns (by reference) the value of the data field at the coordinates (i, j, k) */
        GT_FUNCTION
        value_type& operator()(uint_t i, uint_t j, uint_t k) {
#ifdef __CUDACC__
            assert(_index(i,j,k) < size());
#endif
            return (m_fields[0])[_index(i,j,k)];
        }


        /** @brief returns (by const reference) the value of the data field at the coordinates (i, j, k) */
        GT_FUNCTION
        value_type const & operator()(uint_t i, uint_t j, uint_t k) const {
#ifdef __CUDACC__
            assert(_index(i,j,k) < size());
#endif
            return (m_fields[0])[_index(i,j,k)];
        }

        /** @brief returns the memory access index of the element with coordinate (i,j,k) */
        //note: offset returns a signed int because the layout map indexes are signed short ints
        GT_FUNCTION
        int_t offset(int_t i, int_t j, int_t k) const {
            return  m_strides[1]* layout::template find_val<0,int,0>(i,j,k) +  m_strides[2] * layout::template find_val<1,int,0>(i,j,k) + layout::template find_val<2,int,0>(i,j,k);
        }

        /**@brief returns the size of the data field*/
        GT_FUNCTION
        uint_t const& size() const {
            return m_strides[0];
        }

        /**@brief prints the first values of the field to standard output*/
        void print() const {
            print(std::cout);
        }

        /**@brief prints a single value of the data field given the coordinates*/
        void print_value(uint_t i, uint_t j, uint_t k){ printf("value(%d, %d, %d)=%f, at index %d on the data\n", i, j, k, (m_fields[0])[_index(i, j, k)], _index(i, j, k));}

        static const std::string info_string;


        /**@brief return the stride for a specific coordinate, given the vector of strides
         Coordinates 0,1,2 correspond to i,j,k respectively*/
        template<uint_t Coordinate>
        GT_FUNCTION
        static constexpr uint_t strides(uint_t const* str){
            return (vec_max<typename layout::layout_vector_t>::value < 0) ?0:( layout::template at_<Coordinate>::value == vec_max<typename layout::layout_vector_t>::value ) ? 1 :  str[layout::template at_<Coordinate>::value+1];
        }

        /**@brief printing a portion of the content of the data field*/
        template <typename Stream>
        void print(Stream & stream, uint_t t=0) const {
		stream << " (" << m_strides[1] << "x"
		       << m_strides[2] << "x"
		       << 1 << ")"
		       << std::endl;
		stream << "| j" << std::endl;
		stream << "| j" << std::endl;
		stream << "v j" << std::endl;
		stream << "---> k" << std::endl;

		ushort_t MI=12;
		ushort_t MJ=12;
		ushort_t MK=12;
		for (uint_t i = 0; i < dims<0>(); i += std::max(( uint_t)1,dims<0>()/MI)) {
		    for (uint_t j = 0; j < dims<1>(); j += std::max(( uint_t)1,dims<1>()/MJ)) {
			for (uint_t k = 0; k < dims<2>(); k += std::max(( uint_t)1,dims<1>()/MK))
			{
			    stream << "["
				// << i << ","
				// << j << ","
				// << k << ")"
                               <<  (m_fields[t])[_index(i,j,k)] << "] ";
			}
			stream << std::endl;
		    }
		    stream << std::endl;
		}
		stream << std::endl;
	}

	    /**@brief returning the index of the memory address corresponding to the specified (i,j,k) coordinates.
         This method depends on the strategy used (either naive or blocking). In case of blocking strategy the
        index for temporary storages is computed in the subclass gridtools::host_tmp_storge*/
        GT_FUNCTION
        uint_t _index(uint_t i, uint_t j, uint_t k) const {
            uint_t index;
            if (IsTemporary) {
                index =
                    m_strides[1]
                    * (modulus(layout::template find_val<0,uint_t,0>(i,j,k),layout::template find<0>(m_dims))) +
                    m_strides[2] * modulus(layout::template find_val<1,uint_t,0>(i,j,k),layout::template find<1>(m_dims)) +
                    modulus(layout::template find_val<2,uint_t,0>(i,j,k),layout::template find<2>(m_dims));
            } else {
		index =
		    m_strides[1]
		    * layout::template find_val<0,uint_t,0>(i,j,k) +
		    m_strides[2] * layout::template find_val<1,uint_t,0>(i,j,k) +
		    layout::template find_val<2,uint_t,0>(i,j,k);
	    }
            return index;
        }

        /** @brief method to increment the memory address index by moving forward one step in the given Coordinate direction

	    SFINAE: design pattern used to avoid the compilation of the overloaded method which are not used (and which would produce a compilation error)
*/
        template <uint_t Coordinate>
        GT_FUNCTION
        void increment( uint_t* index/*, typename boost::enable_if_c< (layout::template pos_<Coordinate>::value >= 0) >::type* dummy=0*/){
	    BOOST_STATIC_ASSERT(Coordinate < space_dimensions);
	    //if(layout::template at_<Coordinate>::value>=0)
	    if(layout::template at_< Coordinate >::value >=0)
	    {
		*index += strides<Coordinate>(m_strides);
	    }
        }

        /** @brief method to decrement the memory address index by moving backward one step in the given Coordinate direction */
        template <uint_t Coordinate>
        GT_FUNCTION
        void decrement( uint_t* index/*, typename boost::enable_if_c< (layout::template pos_<Coordinate>::value >= 0) >::type* dummy=0*/){
	    BOOST_STATIC_ASSERT(Coordinate < space_dimensions);
	    //if(layout::template at_<layout::template pos_<Coordinate>::value >::value>=0)
	    // if(layout::template find_val<Coordinate, int_t, -1>(m_strides)>=0)
	    if(layout::template at_<Coordinate>::value >=0)
	    {
		*index-=strides<Coordinate>(m_strides);
	    }
        }

        /** @brief method to increment the memory address index by moving forward a given number of step in the given Coordinate direction */
        template <uint_t Coordinate>
        GT_FUNCTION
	void increment(uint_t const& dimension, uint_t const& /*block*/, uint_t* index/*, typename boost::enable_if_c< (layout::template pos_<Coordinate>::value >= 0) >::type* dummy=0*/){
	    BOOST_STATIC_ASSERT(Coordinate < space_dimensions);
	    // if(layout::template find_val<Coordinate, int_t, -1>(m_strides)>=0)
	    if( layout::template at_< Coordinate >::value >= 0 )
	    {
		*index += strides<Coordinate>(m_strides)*dimension;
	    }
        }

        /** @brief method to decrement the memory address index by moving backward a given number of step in the given Coordinate direction */
        template <uint_t Coordinate>
        GT_FUNCTION
        void decrement(uint_t dimension, uint_t const& /*block*/, uint_t* index){
	    BOOST_STATIC_ASSERT(Coordinate < space_dimensions);
	    if( layout::template at_< Coordinate >::value >= 0 )
	    {
		*index-=strides<Coordinate>(m_strides)*dimension;
	    }
        }

	GT_FUNCTION
        void set_index(uint_t value, uint_t* index){
	    *index=value;
	}

        /**@brief returns the data field*/
        GT_FUNCTION
        pointer_type const& data() const {return (m_fields[0]);}

        /** @brief returns a pointer to the data field*/
        GT_FUNCTION
        typename pointer_type::pointee_t* get_address() const {
            return (m_fields[0]).get();}

        /** @brief returns a const pointer to the data field*/
        GT_FUNCTION
        pointer_type const* fields() const {return &(m_fields[0]);}

        /** @brief returns the dimension fo the field along I*/
        template<ushort_t I>
        GT_FUNCTION
        uint_t dims() const {return m_dims[I];}

        /**@brief returns the storage strides*/
        GT_FUNCTION
        uint_t strides(ushort_t i) const {
            //"you might thing that with m_srides[0] you are accessing a stride,\n
            // but you are indeed accessing the whole storage dimension."
            assert(i!=0);
            return m_strides[i];
        }

    protected:
        bool is_set;
	const char* m_name;
        // static const uint_t m_strides[/*3*/space_dimensions]={( dim1*dim2*dim3 ),( dims[layout::template get<2>()]*dims[layout::template get<1>()]),( dims[layout::template get<2>()] )};
	pointer_type m_fields[field_dimensions];
        uint_t m_dims[space_dimensions];
        uint_t m_strides[space_dimensions];
#ifdef NDEBUG
    private:
	/**@brief noone calls the empty constructor*/
	base_storage();
#else
	/**only for stdcout purposes*/
	base_storage(){}
#endif
    };


    /**@brief Metafunction for computing the coordinate N from the index (not currently used anywhere)
       N=0 is the coordinate with stride 1*/
    template <ushort_t N>
    struct coord_from_index;

    //specializations for each dimension (supposing we have 3)
    template<>
    struct coord_from_index<2>
    {
        static uint_t apply(uint_t index, uint_t* strides){
            printf("the coord from index: tile along %d is %d\n ", 0, strides[2]);
            return index%strides[2];
        }
    };

    template<>
    struct coord_from_index<1>
        {
            static uint_t apply(uint_t index, uint_t* strides){
                printf("the coord from index: tile along %d is %d\n ", 1, strides[1]);
                return (index%strides[1]// tile<N>::value
                        -index% strides[2]);//(index%(K*J)-index%K%base_type::size()
            }
        };


    template<>
    struct coord_from_index<0>
        {
            static uint_t apply(uint_t index, uint_t* strides){
                printf("the coord from index: tile along %d is %d\n ", 2, strides[0]);
                return (index//%strides[0]
                        -index%strides[1]-index% strides[2]);//(index%(K*J)-index%K
            }
        };

    /**@brief print for debugging purposes*/
    struct print_index{
        template <typename BaseStorage>
        GT_FUNCTION
        void operator()(BaseStorage* b ) const {printf("index -> %d, address %lld, 0x%08x \n", b->index(), &b->index(), &b->index());}
    };

    /** @brief storage class containing a buffer of data snapshots

        the goal of this struct is to  implement a cash for the solutions, in order e.g. to ease the finite differencing between the different scalar fields.
    */
    template < typename Storage, ushort_t ExtraWidth>
    struct extend_width : public Storage//, clonable_to_gpu<extend_width<Storage, ExtraWidth> >
    {
        typedef Storage super;
        typedef typename super::pointer_type pointer_type;

        typedef typename super::original_storage original_storage;
        typedef typename super::iterator_type iterator_type;
        typedef typename super::value_type value_type;

        /**@brief default constructor*/
        explicit extend_width(uint_t const& dim1, uint_t const& dim2, uint_t const& dim3 ): super( dim1, dim2, dim3 ) {
        }


        /**@brief destructor: frees the pointers to the data fields */
        virtual ~extend_width(){
	}

	using super::m_fields;

        /**@brief device copy constructor*/
	template <typename T>
        __device__
        extend_width(T const& other)
            : super(other)
            {
                assert(n_width==other.n_width);
            }

        /**@brief copy all the data fields to the GPU*/
        GT_FUNCTION_WARNING
        void copy_data_to_gpu(){
            //the fields are otherwise not copied to the gpu, since they are not inserted in the storage_pointers fusion vector
            for (uint_t i=0; i< super::field_dimensions; ++i)
                m_fields[i].update_gpu();
        }

        /** @brief returns the address to the first element of the current data field (pointed by (m_fields[0]))*/
        GT_FUNCTION
        typename pointer_type::pointee_t* get_address() const {
            return super::get_address();}


        /**
           @brief returns the index (in the array of data snapshots) corresponding to the specified offset
           basically it returns offset unless it is negative or it exceeds the size of the internal array of snapshots. In the latter case it returns offset modulo the size of the array.
           In the former case it returns the array size's complement of -offset.
         */
        GT_FUNCTION
        static constexpr ushort_t get_index (short_t const& offset) {
            return (offset+n_width)%n_width;
        }

        /** @brief returns the address of the first element of the specified data field
            The data field to be accessed is identified given an offset, which is the index of the local array of snapshots.
         */
        GT_FUNCTION
        typename pointer_type::pointee_t* get_address(short_t offset) const {
            return m_fields[get_index(offset)].get();}
        GT_FUNCTION
        pointer_type const& get_field(int index) const {return m_fields[index];};

        /**@brief swaps the argument with the last data snapshots*/
        GT_FUNCTION
        void swap(pointer_type & field){
            //the integration takes ownership over all the pointers?
            //cycle in a ring
            pointer_type swap(m_fields[super::field_dimensions-1]);
            m_fields[super::field_dimensions-1]=field;
            field = swap;
        }

        /**@brief adds a given data field at the front of the buffer
           \param field the pointer to the input data field
           NOTE: better to shift all the pointers in the array, because we do this seldomly, so that we don't need to keep another indirection when accessing the storage ("stateless" buffer)
         */
        GT_FUNCTION
        void push_front( pointer_type& field, uint_t const& from=(uint_t)0, uint_t const& to=(uint_t)(n_width)){
	    //Too many shaphots pushed! exceeding the buffer width allocated of the storage.
	    assert(!m_fields[to-1].get());

            //cycle in a ring: better to shift all the pointers, so that we don't need to keep another indirection when accessing the storage (stateless buffer)
            for(uint_t i=from+1;i<to;i++) m_fields[i]=m_fields[i-1];
            m_fields[from]=(field);
        }

        //the time integration takes ownership over all the pointers?
	/**TODO code repetition*/
        GT_FUNCTION
        void advance(uint_t from=(uint_t)0, uint_t to=(uint_t)(n_width)){
            pointer_type tmp(m_fields[to-1]);
            for(uint_t i=from+1;i<to;i++) m_fields[i]=m_fields[i-1];
            m_fields[from]=tmp;
        }

        GT_FUNCTION
        pointer_type const*  fields(){
	    return super::fields();
	}

	/**@brief printing the first values of all the snapshots contained in the discrete field*/
        void print() {
            print(std::cout);
        }

	/**@brief printing the first values of all the snapshots contained in the discrete field, given the output stream*/
        template <typename Stream>
        void print(Stream & stream) {
	    for (ushort_t t=0; t<super::field_dimensions; ++t)
	    {
		stream<<" Component: "<< t+1<<std::endl;
		original_storage::print(stream, t);
	    }
        }

        static const ushort_t n_width = ExtraWidth+1;

        //for stdcout purposes
        explicit extend_width(){}

    };

    /**@brief specialization: if the width extension is 0 we fall back on the base storage*/
    template < typename Storage>
    struct extend_width<Storage, 0> : public Storage//, clonable_to_gpu<extend_width<Storage, 0> > //  : public Storage
    {
        typedef typename Storage::basic_type basic_type;
        typedef typename Storage::original_storage original_storage;

        /**@brief default constructor*/
        explicit extend_width(uint_t const& dim1, uint_t const& dim2, uint_t const& dim3 ): Storage( dim1, dim2, dim3 ) {
        }

        /**@brief destructor: frees the pointers to the data fields */
        virtual ~extend_width(){
	}

        static const ushort_t n_width = Storage::n_width;

	/**@brief device copy constructor*/
        __device__
        extend_width(extend_width const& other)
            : Storage(other)
            {}

    private:
        //for stdcout purposes
        explicit extend_width(){}

    };

#ifdef CXX11_ENABLED

    /** @brief traits class defining some useful compile-time counters
    */
    template < typename First, typename  ...  StorageExtended>
    struct dimension_extension_traits// : public dimension_extension_traits<StorageExtended ... >
    {
        //total buffer size
        static const uint_t n_fields=First::n_width + dimension_extension_traits<StorageExtended  ...  >::n_fields ;
        //the buffer size of the current field (i.e. the total number of snapshots)
        static const uint_t n_width=First::n_width;
        //the number of dimensions (i.e. the number of different fields)
        static const uint_t n_dimensions=  dimension_extension_traits<StorageExtended  ...  >::n_dimensions  +1 ;
        //the current field extension
        // typedef extend_width<First, n_fields>  type;
	//n_fields-1 because the extend_width takes the EXTRA width as argument, not the total width.
        typedef extend_width<First, n_fields-1>  type;
        // typedef First type;
        typedef dimension_extension_traits<StorageExtended ... > super;
   };

    /**@brief fallback in case the snapshot we try to access exceeds the width diemnsion assigned to a discrete scalar field*/
    struct dimension_extension_null{
        static const uint_t n_fields=0;
        static const uint_t n_width=0;
        static const uint_t n_dimensions=0;
        //typedef extend_width<First, n_fields>  type;
        typedef struct error_index_too_large1{} type;
        typedef struct error_index_too_large2{} super;
    };

/**@brief template specialization at the end of the recustion.*/
    template < typename First>
    struct dimension_extension_traits<First>  {
        //total number of dimensions
        static const uint_t n_fields=First::n_width;
        static const uint_t n_width=First::n_width;
        static const uint_t n_dimensions= 1 ;
        //typedef extend_width<First, n_fields>  type;
        typedef First type;
        typedef dimension_extension_null super;
     };

    /**@brief metafunction to access a type sequence at a given position, numeration from 0

       The types in the sequence must define a 'super' type. Can be seen as a compile-time equivalent of a linked-list.
     */
    template<int_t ID, typename Sequence>
    struct access{
	BOOST_STATIC_ASSERT(ID>0);
	//BOOST_STATIC_ASSERT(ID<=Sequence::n_fields);
        typedef typename access<ID-1, typename Sequence::super>::type type;
    };

    /**@brief template specialization to stop the recursion*/
    template<typename Sequence>
    struct access<0, Sequence>{
        typedef Sequence type;
    };

    /**recursively advance the ODE finite difference for all the field dimensions*/
    template<short_t Dimension>
    struct advance_recursive{
	template<typename This>
	void apply(This* t){
	    t->template advance<Dimension>();
	    advance_recursive<Dimension-1>::apply(t);
	}
    };

    /**template specialization to stop the recursion*/
    template<>
    struct advance_recursive<0>{
	template<typename This>
	void apply(This* t){
	    t->template advance<0>();
	}
    };

    /**@brief implements the discrete vector field structure

     It is a collection of arbitrary length discrete scalar fields.
    */
    template <typename First,  typename  ...  StorageExtended>
    struct extend_dim : public dimension_extension_traits<First, StorageExtended ... >::type, clonable_to_gpu<extend_dim<First, StorageExtended ... > >
    {
        typedef typename dimension_extension_traits<First, StorageExtended ... >::type super;
        typedef dimension_extension_traits<First, StorageExtended ...  > traits;
        typedef typename super::pointer_type pointer_type;
        typedef typename  super::basic_type basic_type;
        typedef typename super::original_storage original_storage;
        //inheriting constructors
        //using typename super::extend_width;
	static const uint n_width=sizeof...(StorageExtended)+1;

        extend_dim(  const uint& d1, const uint& d2, const uint& d3 )
            : super(d1, d2, d3)
            {
	    }

        __device__
        extend_dim( extend_dim const& other )
            : super(other)
            {}

        /**@brief destructor: frees the pointers to the data fields */
        virtual ~extend_dim(){
	}

        /**@brief pushes a given data field at the front of the buffer for a specific dimension
           \param field the pointer to the input data field
	   \tparam dimension specifies which field dimension we want to access
         */
        template<uint_t dimension=1>
        GT_FUNCTION
        void push_front( pointer_type& field ){//copy constructor
            //cycle in a ring: better to shift all the pointers, so that we don't need to keep another indirection when accessing the storage (stateless storage)

            BOOST_STATIC_ASSERT(n_width>dimension);
            BOOST_STATIC_ASSERT(dimension<=traits::n_fields);
            uint_t const indexFrom=access<n_width-dimension, traits>::type::n_fields;
            uint_t const indexTo=access<n_width-dimension-1, traits>::type::n_fields;

	    super::push_front(std::forward<pointer_type&>(field), indexFrom, indexTo);
        }

	/**@brief Pushes the given storage as the first snapshot at the specified field dimension*/
        template<uint_t dimension=1>
        GT_FUNCTION
        void push_front( pointer_type& field, typename super::value_type const& value ){//copy constructor
	    for (uint_t i=0; i<super::size(); ++i)
	     	field[i]=value;
	    push_front<dimension>(field);
	}

	/**@biref sets the given storage as the nth snapshot of a specific field dimension

	   \tparam field_dim the given field dimenisons
	   \tparam snapshot the snapshot of dimension field_dim to be set
	   \param field the input storage
	 */
	template<short_t field_dim, short_t snapshot>
	void set( pointer_type& field)
	    {
		super::m_fields[access<n_width-(field_dim), traits>::type::n_fields + snapshot]=field;
	    }

	/**@biref sets the given storage as the nth snapshot of a specific field dimension and initialize the storage with an input constant value

	   \tparam field_dim the given field dimenisons
	   \tparam snapshot the snapshot of dimension field_dim to be set
	   \param field the input storage
	   \param val the initializer value
	 */
	template<short_t field_dim, short_t snapshot>
	void set( pointer_type& field, typename super::value_type const& val)
	    {
		for (uint_t i=0; i<super::size(); ++i)
		    field[i]=val;
		set<field_dim, snapshot>(field);
	    }

	/**@biref sets the given storage as the nth snapshot of a specific field dimension and initialize the storage with an input lambda function

	   \tparam field_dim the given field dimenisons
	   \tparam snapshot the snapshot of dimension field_dim to be set
	   \param field the input storage
	   \param lambda the initializer function
	 */
	template<short_t field_dim, short_t snapshot>
	void set( pointer_type& field, typename super::value_type (*lambda)(uint_t, uint_t, uint_t))
	    {
		for (uint_t i=0; i<this->m_dims[0]; ++i)
		    for (uint_t j=0; j<this->m_dims[1]; ++j)
			for (uint_t k=0; k<this->m_dims[2]; ++k)
			    (super::m_fields[access<n_width-(field_dim), traits>::type::n_fields + snapshot])[super::_index(i,j,k)]=lambda(i, j, k);
	    }

	/**@biref ODE advancing for a single dimension

	   it advances the supposed finite difference scheme of one step for a specific field dimension
	   \tparam dimension the dimension to be advanced
	   \param offset the number of steps to advance
	 */
        template<uint_t dimension=1>
        GT_FUNCTION
        void advance(){
            BOOST_STATIC_ASSERT(dimension<traits::n_dimensions);
            uint_t const indexFrom=access<dimension, traits>::type::n_fields;
            uint_t const indexTo=access<dimension-1, traits>::type::n_fields;

            super::advance(indexFrom, indexTo);
        }


	/**@biref ODE advancing for a single dimension

	   it advances the supposed finite difference scheme of one step for a specific field dimension
	   \tparam dimension the dimension to be advanced
	   \param offset the number of steps to advance
	 */
        GT_FUNCTION
        void advance_all(){
            advance_recursive<n_width>::apply(const_cast<extend_dim*>(this));
        }

        //for stdcout purposes
        explicit extend_dim(){}
    };

    /**@brief Convenient syntactic sugar for specifying an extended-dimension with extended-width storages, where each dimension has arbitrary size 'Number'.
       Annoyngly enough does not work with CUDA 6.5
     */
#if !defined(__CUDACC__)
    template< class Storage, uint_t ... Number >
			  struct extend{
	typedef extend_dim< extend_width<base_storage<Storage::backend, typename Storage::value_type, typename  Storage::layout, Storage::is_temporary, accumulate(add(), ((uint_t)Number+1) ... )>, Number> ... > type;
    };
#endif

#endif //CXX11_ENABLED

/** \addtogroup specializations Specializations
    Partial specializations
    @{
 */
    template < enumtype::backend B, typename ValueType, typename Layout, bool IsTemporary, short_t Dim
               >
    const std::string base_storage<B , ValueType, Layout, IsTemporary, Dim
                                   >::info_string=boost::lexical_cast<std::string>("-1");

    template <enumtype::backend B, typename ValueType, typename Y, short_t Dim>
    struct is_temporary_storage<base_storage<B,ValueType,Y,false, Dim>*& >
        : boost::false_type
    {};

    template <enumtype::backend X, typename ValueType, typename Y, short_t Dim>
    struct is_temporary_storage<base_storage<X,ValueType,Y,true, Dim>*& >
        : boost::true_type
    {};

    template <enumtype::backend X, typename ValueType, typename Y, short_t Dim>
    struct is_temporary_storage<base_storage<X,ValueType,Y,false, Dim>* >
        : boost::false_type
    {};

    template <enumtype::backend X, typename ValueType, typename Y, short_t Dim>
    struct is_temporary_storage<base_storage<X,ValueType,Y,true, Dim>* >
        : boost::true_type
    {};

    template <enumtype::backend X, typename ValueType, typename Y, short_t Dim>
    struct is_temporary_storage<base_storage<X,ValueType,Y,false, Dim> >
        : boost::false_type
    {};

    template <enumtype::backend X, typename ValueType, typename Y, short_t Dim>
    struct is_temporary_storage<base_storage<X,ValueType,Y,true, Dim> >
        : boost::true_type
    {};

    template <  template <typename T> class  Decorator, typename BaseType>
    struct is_temporary_storage<Decorator< BaseType > > : is_temporary_storage< typename BaseType::basic_type >
    {};
    template <  template <typename T> class Decorator, typename BaseType>
    struct is_temporary_storage<Decorator< BaseType >* > : is_temporary_storage< typename BaseType::basic_type* >
    {};
    template <  template <typename T> class Decorator, typename BaseType>
    struct is_temporary_storage<Decorator< BaseType >& > : is_temporary_storage< typename BaseType::basic_type& >
    {};
    template <  template <typename T> class Decorator, typename BaseType>
    struct is_temporary_storage<Decorator< BaseType >*& > : is_temporary_storage< typename BaseType::basic_type*& >
    {};

#ifdef CXX11_ENABLED
    //Decorator is the extend
    template <template <typename ... T> class Decorator, typename First, typename ... BaseType>
    struct is_temporary_storage<Decorator<First, BaseType...> > : is_temporary_storage< typename First::basic_type >
    {};

    //Decorator is the extend
    template <template <typename ... T> class Decorator, typename First, typename ... BaseType>
    struct is_temporary_storage<Decorator<First, BaseType...>* > : is_temporary_storage< typename First::basic_type* >
    {};

    //Decorator is the extend
    template <template <typename ... T> class Decorator, typename First, typename ... BaseType>
    struct is_temporary_storage<Decorator<First, BaseType...>& > : is_temporary_storage< typename First::basic_type& >
    {};

    //Decorator is the extend
    template <template <typename ... T> class Decorator, typename First, typename ... BaseType>
    struct is_temporary_storage<Decorator<First, BaseType...>*& > : is_temporary_storage< typename First::basic_type*& >
    {};
#endif //CXX11_ENABLED
/**@}*/
    template <enumtype::backend Backend, typename T, typename U, bool B>
    std::ostream& operator<<(std::ostream &s, base_storage<Backend,T,U, B> ) {
        return s << "base_storage <T,U," << " " << std::boolalpha << B << "> ";
    }

#ifdef CXX11_ENABLED
    template <typename ... T>
    std::ostream& operator<<(std::ostream &s, extend_dim< T... > ) {
        return s << "extend_dim storage" ;
    }
#endif

} //namespace gridtools
