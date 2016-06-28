#ifndef _LAYOUT_MAP_H_
#define _LAYOUT_MAP_H_

#include <gridtools.hpp>
#include <boost/static_assert.hpp>
#include <boost/mpl/vector_c.hpp>
#include <boost/mpl/int.hpp>
#include <boost/mpl/at.hpp>
#include "../common/gt_assert.hpp"
#include "../common/host_device.hpp"
#include "../common/defs.hpp"
#include "../common/array.hpp"
#include "stencil-composition/accessor_fwd.hpp"
#include "generic_metafunctions/gt_get.hpp"
#include "common/generic_metafunctions/is_variadic_pack_of.hpp"
#include "generic_metafunctions/gt_expand.hpp"
#include "generic_metafunctions/accumulate.hpp"
#include "generic_metafunctions/gt_get.hpp"

/**
   @file
   @brief definifion of the data layout
   Here are defined the classes select_s and layout_map.
*/
namespace gridtools {

    //fwd decl
    template <typename T>
    struct is_layout_map;

/**
   @struct
   @brief Used as template argument in the storage.
   In particular in the \ref gridtools::base_storage class it regulate memory access order, defined at compile-time, by leaving the interface unchanged.
*/
    namespace _impl {

        template <int index>
        static int __get(int i) {
            return -1;
        }

        template <int index, int first, int... Vals>
        static int __get(int i) {
            if (i == index) {
                return first;
            } else {
                return __get<index+1, Vals...>(i);
            }
        }

    }//namespace _impl


    // forward declarations
    template < ushort_t ID, enumtype::intend Intend, typename Extent, ushort_t Number>
    struct accessor;

    template <typename ArgType, typename ... Pair>
    struct accessor_mixed;

    //template arguments type checking
    template <typename T>
    struct is_arg_tuple : boost::false_type {};

    template < ushort_t ID, enumtype::intend Intend, typename Extent, ushort_t Number>
    struct is_arg_tuple<accessor<ID, Intend, Extent, Number> > : boost::true_type{};

    template <typename ArgType, typename ... Pair>
    struct is_arg_tuple<accessor_mixed<ArgType, Pair ... > > : boost::true_type {};


    /**
       Layout maps are simple sequences of integers specified
       statically. The specification happens as

       \code
       gridtools::layout_map<a,b,c>
       \endcode

       where a, b, and c are integer static constants. To access the
       elements of this sequences the user should call the static method

       \code
       ::at<I>()
       \endcode

       For instance:
       \code
       gridtools::layout_map<3,4,1,5>::at<2> == 1
       gridtools::layout_map<3,4,1,5>::at<0> == 3
       etc.
       \endcode
    */
    template <short_t ... Args>
    struct layout_map{
        static constexpr ushort_t length=sizeof...(Args);
        static const constexpr short_t layout_vector[sizeof...(Args)]={Args...};
        typedef boost::mpl::vector_c<short_t, Args...> layout_vector_t;
        /* BOOST_STATIC_ASSERT(s); */

        template <class Layout >
        struct append{

            GRIDTOOLS_STATIC_ASSERT( is_layout_map<Layout>::value, "internal error" );

            typedef typename boost::mpl::fold<
                typename Layout::layout_vector_t,
                layout_map<Args ... >,
                layout_map<Args ..., boost::mpl::plus<boost::mpl::_2, static_ushort<length> >::value > >::type type;
        };

        /** This function returns the value in the map that is stored at
            position 'I', where 'I' is passed in input as template
            argument.

            \tparam I The index to be queried
        */
        template <ushort_t I>
        GT_FUNCTION
        static constexpr short_t at() {
            BOOST_STATIC_ASSERT( I<length );
            return layout_vector[ I ];
        }


        template <typename T>
        struct remove_refref;

        template <typename T>
        struct remove_refref<T&&> {
            using type=T;
        };

#ifndef __CUDACC__
        /** Given a parameter pack of values and a static index, the function
            returns the reference to the value in the position indicated
            at position 'I' in the map.

            \code
            gridtools::layout_map<1,2,0>::select<1>(a,b,c) == c
            \endcode

            \tparam I Index to be queried
            \tparam T Sequence of types
            \param[in] args Values from where to select the element  (length must be equal to the length of the layout_map length)
        */
        template <ushort_t I, typename ... T>
        GT_FUNCTION
        static auto constexpr select(T & ... args) -> typename remove_refref<decltype(std::template get<layout_vector[I]>(std::make_tuple(args ...)))>::type {

            GRIDTOOLS_STATIC_ASSERT((is_variadic_pack_of(boost::is_integral<T>::type::value ...)), "wrong type");
            return  gt_get<layout_vector[I]>::apply( args ... );

        }
#else //problem determining of the return type with NVCC
        template <ushort_t I, typename First, typename ... T>
        GT_FUNCTION
        static First
        constexpr
        select(First & f, T & ... args) {
            GRIDTOOLS_STATIC_ASSERT((boost::is_integral<First>::type::value &&
                                     is_variadic_pack_of(boost::is_integral<T>::type::value ...)), "wrong type");
            return  gt_get<boost::mpl::at_c<layout_vector_t, I>::type::value>::apply( f, args... );
        }
#endif // __CUDACC__

        //returns the dimension corresponding to the given strides (get<0> for stride 1)
        template <ushort_t i>
        GT_FUNCTION
        static constexpr ushort_t get() {
            return layout_vector[i];
        }

        GT_FUNCTION
        short_t constexpr operator[](ushort_t i) const {
            return _impl::__get<0, Args...>(i);
        }

        struct transform_in_type{
            template<ushort_t T>
            struct apply{
                typedef static_ushort<T> type;
            };
        };

        template< ushort_t I, ushort_t T >
        struct predicate{
            typedef typename boost::mpl::bool_<T==I>::type type;
        };


        /** Given a parameter pack of values and a static index I, the function
            returns the reference to the element whose position
            corresponds to the position of 'I' in the map.

            \code
            gridtools::layout_map<2,0,1>::find<1>(a,b,c) == c
            \endcode

            \tparam I Index to be searched in the map
            \tparam[in] Indices List of values where element is selected
            \param[in] indices  (length must be equal to the length of the layout_map length)
        */
        template <ushort_t I, typename First, typename... Indices>
        GT_FUNCTION
        static constexpr First
        find(First const& first_, Indices const& ... indices) {
            GRIDTOOLS_STATIC_ASSERT(sizeof...(Indices)+1<=length, "Too many arguments");

            return gt_get<pos_<I>::value>::apply(first_, indices...);
        }


        /**@brief traits class allowing the lazy static analysis

           hiding a type whithin a templated struct disables its type deduction, so that when a compile-time branch (e.g. using boost::mpl::eval_if) is not taken, it is also not compiled.
           The following struct defines a subclass with a templated method which returns a given element in a tuple.
        */
        template<ushort_t I, typename Int>
        struct tied_type
        {
            struct type{
                template<typename ... Indices>
                GT_FUNCTION
                static constexpr const Int value(Indices const& ... indices){
                    GRIDTOOLS_STATIC_ASSERT((accumulate(logical_and(), boost::is_integral<Indices>::type::value ...)), "wrong type");
                    return gt_get< pos_<I>::value>::apply( indices ...);
                    //std::get< pos_<I>::value >(std::make_tuple(indices...));
                }
            };
        };

        /**@brief traits class allowing the lazy static analysis

           hiding a type whithin a templated struct disables its type deduction, so that when a compile-time branch (e.g. using boost::mpl::eval_if) is not taken, it is also not compiled.
           The following struct implements a fallback case, when the index we are looking for in the layout_map is not present. It simply returns the default parameter passed in as template argument.
        */
        template<typename Int, Int Default>
        struct identity
        {
            struct type{
                template<typename ... Indices>
                GT_FUNCTION
                static constexpr Int value(Indices ... /*indices*/){return Default;}
            };
        };

        /** Given a parameter pack of values and a static index I, the function
            returns the value of the element whose position
            corresponds to the position of 'I' in the map. If the
            value is not found a default value is returned, which is
            passed as template parameter. It works for intergal types.

            Default value is picked by default if C++11 is anabled,
            otherwise it has to be provided.

            \code
            gridtools::layout_map<2,0,1>::find_val<1,type,default>(a,b,c) == c
            \endcode

            \tparam I Index to be searched in the map
            \tparam Default_Val Default value returned if the find is not successful
            \tparam[in] Indices List of argument where to return the found value
            \param[in] indices List of values (length must be equal to the length of the layout_map length)
        */
        template <ushort_t I, typename T, T DefaultVal, typename ... Indices, typename First,  typename boost::enable_if<boost::is_integral<T>, int>::type=0>
        GT_FUNCTION
        static constexpr T find_val(First const& first, Indices const& ... indices) {
            static_assert(sizeof...(Indices)<=length, "Too many arguments");
            //lazy template instantiation
            typedef typename boost::mpl::eval_if_c< (pos_<I>::value >= sizeof ... (Indices) +1 ),
                identity<T, DefaultVal>
                ,
                tied_type<I, T> >::type type;

            GRIDTOOLS_STATIC_ASSERT((boost::is_integral<First>::type::value), "wrong type");

            return type::value(first, indices...);
        }



        /** @brief finds the value of the argument vector in correspondance of dimension I according to this layout
            \tparam I dimension (0->i, 1->j, 2->k, ...)
            \tparam T type of the return value
            \tparam DefaultVal default value return when the dimension I does not exist
            \tparam Indices type of the indices passed as argument
            \param indices argument vector of indices
        */
        template <ushort_t I, typename T, T DefaultVal, typename Indices>
        GT_FUNCTION
        static constexpr Indices
        find_val(Indices const * indices) {
            return (pos_<I>::value >= length ) ?
                DefaultVal
                :
                indices[pos_<I>::value];
        }

        template<int_t Index, int_t NDim>
        struct offset_tuple;

        /** Given a tuple and a static index I, the function
            returns the value of the element in the tuple whose position
            corresponds to the position of 'I' in the map. If the
            value is not found a default value is returned, which is
            passed as template parameter. It works for intergal types.

            \code
            tuple=arg_type(a,b,c);
            gridtools::layout_map<2,0,1>::find_val<1,type,default>(tuple) == c
            \endcode

            \tparam I Index to be searched in the map
            \tparam Default_Val Default value returned if the find is not successful
            \tparam[in] Indices List of argument where to return the found value
            \param[in] indices List of values (length must be equal to the length of the layout_map length)
        */
        template <ushort_t I, typename T, T DefaultVal, typename Accessor, typename boost::enable_if<is_arg_tuple< Accessor >, int>::type=0 >
        GT_FUNCTION
        static constexpr T find_val(Accessor const& indices) {

            // GRIDTOOLS_STATIC_ASSERT( length<=2 || length <= Accessor::n_dim, "pedantic check: an accessor's dimension is smaller than the corresponding storage space dimension");
            GRIDTOOLS_STATIC_ASSERT((is_accessor<Accessor>::value), "the find_val method is used with tuples of type other than accessor");
            GRIDTOOLS_STATIC_ASSERT((Accessor::n_dim-pos_<I>::value-1>=0), "write a message here");
            return ((pos_<I>::value >= length)) ?
                DefaultVal
                :
                indices.template get<Accessor::n_dim-pos_<I>::value-1>();
            //this calls arg_decorator::get
        }

        template <ushort_t I, typename MplVector>
        GT_FUNCTION
        static constexpr uint_t find() {
            static_assert(I<length, "Index out of bound");
            return boost::mpl::at_c< MplVector, pos_<I>::value>::type::value;
        }

        /** Given a tuple of values and a static index I, the function
            returns the reference to the element whose position
            corresponds to the position of 'I' in the map.

            \code
            a[0] = a; a[1] = b; a[3] = c;
            gridtools::layout_map<2,0,1>::find<1>(a) == c
            \endcode

            \tparam I Index to be searched in the map
            \tparam T Types of elements
            \param[in] a Pointer to a region with the elements to match
        */
        template <ushort_t I, typename T>
        GT_FUNCTION
        static uint_t find(const T* indices) {
            static_assert(I<length, "Index out of bound");
            return indices[pos_<I>::value];
        }


        template <ushort_t I>
        struct at_ {
#ifdef PEDANTIC
            static_assert(I<length, "Index out of bound: accessing an object with a layout map (a storage) using too many indices.");
#endif
            static const short_t value = I<length ? layout_vector[I] : -1;
        };

        template <ushort_t I, short_t DefaultVal>
        struct at_default {
            static_assert(I<length, "Index out of bound");
            static const short_t _value = layout_vector[I];
            static const short_t value = (_value<0)?DefaultVal:_value;
        };

        // Gives the position at which I is. e.g., I want to know which is the stride of i (0)?
        //then if pos_<0> is 0, then the index i has stride 1, and so on ...
        template <ushort_t I>
        struct pos_ {
            static_assert(I<=length, "Index out of bound");
            static_assert(I>=0, "Accessing a negative dimension");

            template <ushort_t X, bool IsHere>
            struct _find_pos
            {
                static constexpr ushort_t value = _find_pos<X+1, boost::mpl::at_c<layout_vector_t,  (X+1>=length)?X:X+1>::type::value == I>::value;
            };

            template <ushort_t X>
            struct _find_pos<X, true> {
                static constexpr ushort_t value = X;
            };


            template <bool IsHere>
            struct _find_pos<length+1, IsHere> {
                static constexpr short_t value=-2;
            };

            // stops the recursion and returns a nonsense value
            template <bool IsHere>
            struct _find_pos<length, IsHere> {
                static constexpr ushort_t value = ~ushort_t();
            };

            static constexpr ushort_t value = _find_pos<0, boost::mpl::at_c<layout_vector_t, 0>::type::value == I>::value;

        };

    };

    template<short_t ... Args>
    template <ushort_t I>
    const short_t layout_map<Args...>::at_<I>::value;

    template <short_t ... Args>
    constexpr const short_t layout_map<Args ... >::layout_vector[sizeof...(Args)];


    template <typename layout> struct is_layout_map : boost::mpl::false_{};
    template <short_t ... Args> struct is_layout_map<layout_map<Args...> > : boost::mpl::true_{};

    template < typename Map, ushort_t Pre, ushort_t Post >
    struct sub_map{
        typedef typename gt_expand<typename Map::layout_vector_t, layout_map, Pre, Post>::type type;
    };

    template <typename LM>
    struct reverse_map;

    template <short_t I1, short_t I2>
    struct reverse_map<layout_map<I1, I2> > {
        typedef layout_map<I2,I1> type;
    };

    template <short_t I1, short_t I2, short_t I3>
    struct reverse_map<layout_map<I1, I2,I3> > {
        template <short_t I, short_t Dummy>
        struct new_value;

        template <short_t Dummy>
        struct new_value<0, Dummy> {
            static const short_t value = 2;
        };

        template <short_t Dummy>
        struct new_value<1, Dummy> {
            static const short_t value = 1;
        };

        template <short_t Dummy>
        struct new_value<2, Dummy> {
            static const short_t value = 0;
        };

        typedef layout_map<new_value<I1,0>::value, new_value<I2,0>::value, new_value<I3,0>::value > type;
    };

    template <typename DATALO, typename PROCLO>
    struct layout_transform;

    template <short_t I1, short_t I2, short_t P1, short_t P2>
    struct layout_transform<layout_map<I1,I2>, layout_map<P1,P2> > {
        typedef layout_map<I1,I2> L1;
        typedef layout_map<P1,P2> L2;

        static const short_t N1 = boost::mpl::at_c<typename L1::layout_vector_t, P1>::type::value;
        static const short_t N2 = boost::mpl::at_c<typename L1::layout_vector_t, P2>::type::value;

        typedef layout_map<N1,N2> type;

    };

    template <short_t I1, short_t I2, short_t I3, short_t P1, short_t P2, short_t P3>
    struct layout_transform<layout_map<I1,I2,I3>, layout_map<P1,P2,P3> > {
        typedef layout_map<I1,I2,I3> L1;
        typedef layout_map<P1,P2,P3> L2;

        static const short_t N1 = boost::mpl::at_c<typename L1::layout_vector_t, P1>::type::value;
        static const short_t N2 = boost::mpl::at_c<typename L1::layout_vector_t, P2>::type::value;
        static const short_t N3 = boost::mpl::at_c<typename L1::layout_vector_t, P3>::type::value;

        typedef layout_map<N1,N2,N3> type;

    };

    template <short_t D>
    struct default_layout_map;

    template <>
    struct default_layout_map<1> {
        typedef layout_map<0> type;
    };

    template <>
    struct default_layout_map<2> {
        typedef layout_map<0,1> type;
    };

    template <>
    struct default_layout_map<3> {
        typedef layout_map<0,1,2> type;
    };

    template <>
    struct default_layout_map<4> {
        typedef layout_map<0,1,2,3> type;
    };

} // namespace gridtools



#endif
