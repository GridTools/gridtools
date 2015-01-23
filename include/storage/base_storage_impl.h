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

	// //forward declaration
	// template < typename Storage, ushort_t ExtraWidth>
	// struct extend_width;

// 	/** @brief traits class defining some useful compile-time counters
// 	 */
// 	template < typename First, typename  ...  StorageExtended>
// 	struct dimension_extension_traits// : public dimension_extension_traits<StorageExtended ... >
// 	{
// 	    //total number of snapshots in the discretized field
// 	    static const uint_t n_fields=First::n_width + dimension_extension_traits<StorageExtended  ...  >::n_fields ;
// 	    //the buffer size of the current dimension (i.e. the number of snapshots in one dimension)
// 	    static const uint_t n_width=First::n_width;
// 	    //the number of dimensions (i.e. the number of different fields)
// 	    static const uint_t n_dimensions=  dimension_extension_traits<StorageExtended  ...  >::n_dimensions  +1 ;
// 	    //the current field extension
// 	    //n_fields-1 because the extend_width takes the EXTRA width as argument, not the total width.
// 	    typedef extend_width<First, n_fields-1>  type;
// 	    // typedef First type;
// 	    typedef dimension_extension_traits<StorageExtended ... > super;
// 	};

// 	/**@brief fallback in case the snapshot we try to access exceeds the width diemnsion assigned to a discrete scalar field*/
// 	struct dimension_extension_null{
// 	    static const uint_t n_fields=0;
// 	    static const uint_t n_width=0;
// 	    static const uint_t n_dimensions=0;
// 	    typedef struct error_index_too_large1{} type;
// 	    typedef struct error_index_too_large2{} super;
// 	};

// /**@brief template specialization at the end of the recustion.*/
// 	template < typename First>
// 	struct dimension_extension_traits<First>  {
// 	    static const uint_t n_fields=First::n_width;
// 	    static const uint_t n_width=First::n_width;
// 	    static const uint_t n_dimensions= 1 ;
// 	    typedef First type;
// 	    typedef dimension_extension_null super;
// 	};


	/**@brief recursively advance the ODE finite difference for all the field dimensions*/
	template<short_t Dimension>
	struct advance_recursive{
	    template<typename This>
	    void apply(This* t){
		t->template advance<Dimension>();
		advance_recursive<Dimension-1>::apply(t);
	    }
	};

	/**@brief template specialization to stop the recursion*/
	template<>
	struct advance_recursive<0>{
	    template<typename This>
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

#ifdef __CUDACC__
            template < typename T, typename U, bool B, ushort_t FieldsDimensions
                       >
            GT_FUNCTION_WARNING
            void operator()(base_storage<enumtype::Cuda,T,U,B, FieldsDimensions
                            > *& s) const {
                printf("CIAO POINTER %X\n", s);
            }
#endif


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
