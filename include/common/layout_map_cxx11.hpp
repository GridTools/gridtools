/*
   Copyright 2016 GridTools Consortium

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/
#pragma once

#include <gridtools.hpp>

#include "generic_metafunctions/all_integrals.hpp"
#include "offset_tuple.hpp"
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

    // fwd decl
    template < typename T >
    struct is_vector_accessor;

    template < typename T >
    struct is_layout_map;

    /**
       @struct
       @brief Used as template argument in the storage.
       In particular in the \ref gridtools::base_storage class it regulate memory access order, defined at compile-time,
       by
       leaving the interface unchanged.
    */
    namespace _impl {

        template < int index >
        GT_FUNCTION constexpr static int __get(int i) {
            return -1;
        }

        template < int index, int first, int... Vals >
        GT_FUNCTION constexpr static int __get(int i) {
            return (i == index) ? first : __get< index + 1, Vals... >(i);
        }

    } // namespace _impl

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
    template < short_t... Args >
    struct layout_map {
        static constexpr ushort_t length = sizeof...(Args);
        static const constexpr short_t layout_vector[sizeof...(Args)] = {Args...};
        typedef boost::mpl::vector_c< short_t, Args... > layout_vector_t;

        template < class Layout >
        struct append {

            GRIDTOOLS_STATIC_ASSERT(is_layout_map< Layout >::value, "internal error");

            typedef typename boost::mpl::fold< typename Layout::layout_vector_t,
                layout_map< Args... >,
                layout_map< Args..., boost::mpl::plus< boost::mpl::_2, static_ushort< length > >::value > >::type type;
        };

        /** This function returns the value in the map that is stored at
            position 'I', where 'I' is passed in input as template
            argument.

            \tparam I The index to be queried
        */
        template < ushort_t I >
        GT_FUNCTION static constexpr short_t at() {
            BOOST_STATIC_ASSERT(I < length);
            return layout_vector[I];
        }

        template < typename T >
        struct remove_refref;

        template < typename T >
        struct remove_refref< T && > {
            using type = T;
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
            \param[in] args Values from where to select the element  (length must be equal to the length of the
           layout_map length)
        */
        template < ushort_t I, typename... T >
        GT_FUNCTION static auto constexpr select(T &... args) ->
            typename remove_refref< decltype(std::template get< layout_vector[I] >(std::make_tuple(args...))) >::type {

            GRIDTOOLS_STATIC_ASSERT((is_variadic_pack_of(boost::is_integral< T >::type::value...)), "wrong type");
            return gt_get< layout_vector[I] >::apply(args...);
        }
#else  // problem determining of the return type with NVCC
        template < ushort_t I, typename First, typename... T >
        GT_FUNCTION static First constexpr select(First &f, T &... args) {
            GRIDTOOLS_STATIC_ASSERT((boost::is_integral< First >::type::value &&
                                        is_variadic_pack_of(boost::is_integral< T >::type::value...)),
                "wrong type");
            return gt_get< boost::mpl::at_c< layout_vector_t, I >::type::value >::apply(f, args...);
        }
#endif // __CUDACC__

        // returns the dimension corresponding to the given strides (get<0> for stride 1)
        template < ushort_t i >
        GT_FUNCTION static constexpr ushort_t get() {
            return layout_vector[i];
        }

        GT_FUNCTION
        constexpr short_t operator[](ushort_t i) const { return _impl::__get< 0, Args... >(i); }

        struct transform_in_type {
            template < ushort_t T >
            struct apply {
                typedef static_ushort< T > type;
            };
        };

        template < ushort_t I, ushort_t T >
        struct predicate {
            typedef typename boost::mpl::bool_< T == I >::type type;
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
        template < ushort_t I, typename First, typename... Indices >
        GT_FUNCTION static constexpr First find(First const &first_, Indices const &... indices) {
            GRIDTOOLS_STATIC_ASSERT(sizeof...(Indices) + 1 <= length, "Too many arguments");

            return gt_get< pos_< I >::value >::apply(first_, indices...);
        }

        /* forward declaration*/
        template < ushort_t I >
        struct pos_;

        /**@brief traits class allowing the lazy static analysis

           hiding a type whithin a templated struct disables its type deduction, so that when a compile-time branch
           (e.g. using boost::mpl::eval_if) is not taken, it is also not compiled.
           The following struct defines a subclass with a templated method which returns a given element in a tuple.
        */
        template < ushort_t I, typename Int >
        struct tied_type {
            struct type {
                template < typename... Indices >
                GT_FUNCTION static constexpr const Int value(Indices const &... indices) {
                    GRIDTOOLS_STATIC_ASSERT(
                        (accumulate(logical_and(), boost::is_integral< Indices >::type::value...)), "wrong type");
                    return gt_get< pos_< I >::value >::apply(indices...);
                    // std::get< pos_<I>::value >(std::make_tuple(indices...));
                }
            };
        };

        /**@brief traits class allowing the lazy static analysis

           hiding a type whithin a templated struct disables its type deduction, so that when a compile-time branch
           (e.g. using boost::mpl::eval_if) is not taken, it is also not compiled.
           The following struct implements a fallback case, when the index we are looking for in the layout_map is not
           present. It simply returns the default parameter passed in as template argument.
        */
        template < typename Int, Int Default >
        struct identity {
            struct type {
                template < typename... Indices >
                GT_FUNCTION static constexpr Int value(Indices... /*indices*/) {
                    return Default;
                }
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
        template < ushort_t I,
            typename T,
            T DefaultVal,
            typename... Indices,
            typename First,
            typename Dummy = all_integers< T, First, Indices... > >
        GT_FUNCTION static constexpr T find_val(First const &first, Indices const &... indices) {
            static_assert(sizeof...(Indices) <= length, "Too many arguments");
            // lazy template instantiation
            typedef typename boost::mpl::eval_if_c< (pos_< I >::value >= sizeof...(Indices) + 1),
                identity< T, DefaultVal >,
                tied_type< I, T > >::type type;

            GRIDTOOLS_STATIC_ASSERT((boost::is_integral< First >::type::value), "wrong type");

            return type::value(first, indices...);
        }

        /** @brief finds the value of the argument vector in correspondance of dimension I according to this layout
            \tparam I dimension (0->i, 1->j, 2->k, ...)
            \tparam T type of the return value
            \tparam DefaultVal default value return when the dimension I does not exist
            \tparam Indices type of the indices passed as argument
            \param indices argument vector of indices
        */
        template < ushort_t I, typename T, T DefaultVal, typename Indices >
        GT_FUNCTION static constexpr Indices find_val(Indices const *indices) {
            GRIDTOOLS_STATIC_ASSERT((boost::is_integral< Indices >::type::value), "wrong type");
            return (pos_< I >::value >= length) ? DefaultVal : indices[pos_< I >::value];
        }

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
        template < ushort_t I, typename T, T DefaultVal, typename OffsetTuple >
        GT_FUNCTION static constexpr T find_val(OffsetTuple const &indices) {
            GRIDTOOLS_STATIC_ASSERT((is_offset_tuple< OffsetTuple >::value),
                "the find_val method must be used with tuples of offset_tuple type");
            // GRIDTOOLS_STATIC_ASSERT(length <= OffsetTuple::n_dim, "pedantic check: an accessor's dimension is smaller than the corresponding storage space dimension. Check the functor definition, and the domain_type passed to the make_computation.");
            // GRIDTOOLS_STATIC_ASSERT((OffsetTuple::n_dim-pos_<I>::value-1>=0), "write a message here");
            return ((pos_< I >::value >= length)) ? DefaultVal
                                                  : indices.template get< OffsetTuple::n_dim - pos_< I >::value - 1 >();
            // this calls accessor::get
        }

        template < ushort_t I, typename T, T DefaultVal, typename Value, size_t D >
        GT_FUNCTION static constexpr T find_val(array< Value, D > const &indices) {
            return ((pos_< I >::value >= length)) ? DefaultVal : indices[pos_< I >::value];
        }

        template < ushort_t I, typename MplVector >
        GT_FUNCTION static constexpr uint_t find() {
            static_assert(I < length, "Index out of bound");
            return boost::mpl::at_c< MplVector, pos_< I >::value >::type::value;
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
        template < ushort_t I, typename T >
        GT_FUNCTION static uint_t find(const T *indices) {
            static_assert(I < length, "Index out of bound");
            return indices[pos_< I >::value];
        }

        template < ushort_t I >
        struct at_ {
#ifdef PEDANTIC
            static_assert(I < length,
                "Index out of bound: accessing an object with a layout map (a storage) using too many indices.");
#endif
            static const short_t value = I < length ? layout_vector[I] : -1;
        };

        template < ushort_t I, short_t DefaultVal >
        struct at_default {
            static_assert(I < length, "Index out of bound");
            static const short_t _value = layout_vector[I];
            static const short_t value = (_value < 0) ? DefaultVal : _value;
        };

        // Gives the position at which I is. e.g., I want to know which is the stride of i (0)?
        // then if pos_<0> is 0, then the index i has stride 1, and so on ...
        template < ushort_t I >
        struct pos_ {
            static_assert(I <= length, "Index out of bound");
            static_assert(I >= 0, "Accessing a negative dimension");

            template < ushort_t X, bool IsHere >
            struct _find_pos {
                static constexpr ushort_t value = _find_pos< X + 1,
                    boost::mpl::at_c< layout_vector_t, (X + 1 >= length) ? X : X + 1 >::type::value == I >::value;
            };

            template < ushort_t X >
            struct _find_pos< X, true > {
                static constexpr ushort_t value = X;
            };

            template < bool IsHere >
            struct _find_pos< length + 1, IsHere > {
                static constexpr short_t value = -2;
            };

            // stops the recursion and returns a nonsense value
            template < bool IsHere >
            struct _find_pos< length, IsHere > {
                static constexpr ushort_t value = ~ushort_t();
            };

            static constexpr ushort_t value =
                _find_pos< 0, boost::mpl::at_c< layout_vector_t, 0 >::type::value == I >::value;
        };
    };

    template < short_t... Args >
    template < ushort_t I >
    const short_t layout_map< Args... >::at_< I >::value;

    template < short_t... Args >
    constexpr const short_t layout_map< Args... >::layout_vector[sizeof...(Args)];

    template < typename layout >
    struct is_layout_map : boost::mpl::false_ {};
    template < short_t... Args >
    struct is_layout_map< layout_map< Args... > > : boost::mpl::true_ {};

    template < typename Map, ushort_t Pre, ushort_t Post >
    struct sub_map {
        typedef typename gt_expand< typename Map::layout_vector_t, layout_map, Pre, Post >::type type;
    };
} // namespace gridtools
