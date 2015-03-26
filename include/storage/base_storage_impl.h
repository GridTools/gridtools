#pragma once
#include <boost/lexical_cast.hpp>
#include "../common/basic_utils.h"
#include "../common/gt_assert.h"
#include "../common/is_temporary_storage.h"
//#include "hybrid_pointer.h"
#include <iostream>
#include "accumulate.h"

namespace gridtools{
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

#ifdef CXX11_ENABLED
	/**@brief metafunction to recursively compute the next stride
	   ID goes from space_dimensions-2 to 0
	   MaxIndex is space_dimensions-1
	*/
	template<short_t ID, short_t MaxIndex,  typename Layout>
	struct next_stride{
	    template<typename First, typename ... IntTypes>
            GT_FUNCTION
            static First constexpr apply ( First first, IntTypes ... args){
		return (Layout::template at_<MaxIndex-ID+1>::value==vec_max<typename Layout::layout_vector_t>::value)?0:Layout::template find_val<MaxIndex-ID,short_t,1>(first, args...) * next_stride<ID-1, MaxIndex, Layout>::apply(first, args...);
	    }
	};

	/**@brief template specialization to stop the recursion*/
	template< short_t MaxIndex, typename Layout>
	struct next_stride<0, MaxIndex, Layout>{
	    template<typename First, typename ... IntTypes>
            GT_FUNCTION
            static First constexpr apply(First first, IntTypes ... args){
		return Layout::template find_val<MaxIndex,short_t,1>(first, args...);
	    }
	};

	/**@brief metafunction to recursively compute all the strides, in a generic arbitrary dimensional storage*/
	template<int_t ID, int_t MaxIndex,  typename Layout>
	struct assign_strides{
	    template<typename ... UIntType>
            GT_FUNCTION
            static void apply(uint_t* strides, UIntType ... args){
		BOOST_STATIC_ASSERT(MaxIndex>=ID);
		BOOST_STATIC_ASSERT(ID>=0);
		strides[MaxIndex-ID] = next_stride<ID, MaxIndex, Layout>::apply(args...);
		assign_strides<ID-1, MaxIndex, Layout>::apply(strides, args...);
	    }
	};

	/**@brief specialization to stop the recursion*/
	template< int_t MaxIndex,  typename Layout>
	struct assign_strides<0, MaxIndex, Layout>{
	    template<typename ... UIntType>
            GT_FUNCTION
	    static void apply(uint_t* strides, UIntType ... args){
		BOOST_STATIC_ASSERT(MaxIndex>=0);
		strides[MaxIndex] = next_stride<0, MaxIndex, Layout>::apply(args...);
	    }
	};
#endif

	/**@brief struct to compute the total offset (the sum of the i,j,k indices times their respective strides)
	 */
	template<ushort_t Id, typename Layout>
	struct compute_offset{
	    static const ushort_t space_dimensions = Layout::length;

	    /**interface with an array of coordinates as argument
               \param strides the strides
               \param indices the array of coordinates
            */template<typename IntType>
            GT_FUNCTION
	    static constexpr int_t apply(uint_t const* strides_, IntType* indices_){
		return strides_[space_dimensions-Id]*Layout::template find_val<space_dimensions-Id, int, 0>(indices_)+compute_offset<Id-1, Layout>::apply(strides_, indices_ );
	    }

#ifdef CXX11_ENABLED
            /**interface with an the coordinates as variadic arguments
               \param strides the strides
               \param indices comma-separated list of coordinates
            */
            template<typename ... UInt>
            GT_FUNCTION
            static constexpr int_t apply(uint_t const* strides_, UInt const& ... indices_){
                return strides_[space_dimensions-Id]*Layout::template find_val<space_dimensions-Id, int, 0>(indices_...)+compute_offset<Id-1, Layout>::apply(strides_, indices_... );
            }
#endif
            /**interface with the coordinates as a tuple
               \param strides the strides
               \param indices tuple of coordinates
            */
            template<typename Tuple>
            GT_FUNCTION
            static constexpr int_t apply(uint_t const* strides_, Tuple const&  indices_){
                return (int_t)strides_[space_dimensions-Id]*Layout::template find_val<space_dimensions-Id, int, 0>(indices_)+compute_offset<Id-1, Layout>::apply(strides_, indices_ );
            }

        };

	/**@brief stops the recursion
	 */
	template<typename Layout>
	struct compute_offset<1, Layout>{
	    static const ushort_t space_dimensions = Layout::length;

	    template<typename IntType>
            GT_FUNCTION
	    static int_t apply(uint_t const* /*strides*/, IntType* indices_){
		return Layout::template find_val<space_dimensions-1, int, 0>(indices_);
	    }

#ifdef CXX11_ENABLED
            template<typename ... IntType>
            GT_FUNCTION
            static int_t apply(uint_t const* /*strides*/, IntType const& ... indices_){
                return Layout::template find_val<space_dimensions-1, int, 0>(indices_ ...);
            }
#endif
            /**interface with the coordinates as a tuple
               \param strides the strides
               \param indices tuple of coordinates
            */
            template<typename Tuple>
            GT_FUNCTION
            static int_t apply(uint_t const* /*strides*/, Tuple const&  indices_){
                return Layout::template find_val<space_dimensions-1, int, 0>(indices_);
            }
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

	/**@brief recursively advance the ODE finite difference for all the field dimensions*/
	template<short_t Dimension>
	struct advance_recursive{
	    template<typename This>
            GT_FUNCTION
            void apply(This* t){
		t->template advance<Dimension>();
		advance_recursive<Dimension-1>::apply(t);
	    }
	};

	/**@brief template specialization to stop the recursion*/
	template<>
	struct advance_recursive<0>{
	    template<typename This>
            GT_FUNCTION
	    void apply(This* t){
		t->template advance<0>();
	    }
	};


	// /**@brief Metafunction for computing the coordinate N from the index (not currently used anywhere)
	//    N=0 is the coordinate with stride 1*/
	// template <ushort_t N>
	// struct coord_from_index;

	// //specializations for each dimension (supposing we have 3)
	// template<>
	// struct coord_from_index<2>
	// {
	//     static uint_t apply(uint_t index, uint_t* strides){
	//         printf("the coord from index: tile along %d is %d\n ", 0, strides[2]);
	//         return index%strides[2];
	//     }
	// };

	// template<>
	// struct coord_from_index<1>
	//     {
	//         static uint_t apply(uint_t index, uint_t* strides){
	//             printf("the coord from index: tile along %d is %d\n ", 1, strides[1]);
	//             return (index%strides[1]// tile<N>::value
	//                     -index% strides[2]);//(index%(K*J)-index%K%base_type::size()
	//         }
	//     };


	// template<>
	// struct coord_from_index<0>
	//     {
	//         static uint_t apply(uint_t index, uint_t* strides){
	//             printf("the coord from index: tile along %d is %d\n ", 2, strides[0]);
	//             return (index//%strides[0]
	//                     -index%strides[1]-index% strides[2]);//(index%(K*J)-index%K
	//         }
	//     };

    }//namespace _impl


    namespace _debug{
#ifndef NDEBUG
        struct print_pointer {
            template <typename StorageType>
            GT_FUNCTION_WARNING
            void operator()(StorageType* s) const {
                printf("CIAOOO TATATA %x\n",  s);
            }

// #ifdef __CUDACC__
//             template < typename T, typename U, bool B, ushort_t FieldsDimensions
//                        >
//             GT_FUNCTION_WARNING
//             void operator()(base_storage<T,U,B, FieldsDimensions> *& s) const {
//                 printf("CIAO POINTER %X\n", s);
//             }
// #endif

	    // /**@brief print for debugging purposes*/
	    // struct print_index{
	    //     template <typename BaseStorage>
	    //     GT_FUNCTION
	    //     void operator()(BaseStorage* b ) const {printf("index -> %d, address %lld, 0x%08x \n", b->index(), &b->index(), &b->index());}
	    // };
        };
#endif
    }//namespace _debug
}//namesapace gridtools
