#pragma once
#include <gridtools.hpp>
#include "common/defs.hpp"
#include <boost/lexical_cast.hpp>
#include "../common/is_temporary_storage.hpp"
#include <iostream>
#include "../common/generic_metafunctions/gt_integer_sequence.hpp"
#include "../common/generic_metafunctions/all_integrals.hpp"

namespace gridtools {
    namespace _impl {

#ifdef CXX11_ENABLED
        /**@brief metafunction to recursively compute the next stride
           ID goes from space_dimensions-2 to 0
           MaxIndex is space_dimensions-1
        */
        template < short_t ID, short_t MaxIndex, typename Layout >
        struct next_stride {
            template < typename First, typename... IntTypes >
            GT_FUNCTION static First constexpr apply(First first, IntTypes... args) {
                return Layout::template find_val< MaxIndex - ID, short_t, 1 >(first, args...) *
                       next_stride< ID - 1, MaxIndex, Layout >::apply(first, args...);
            }
        };

        /**@brief template specialization to stop the recursion*/
        template < short_t MaxIndex, typename Layout >
        struct next_stride< 0, MaxIndex, Layout > {
            template < typename First, typename... IntTypes >
            GT_FUNCTION static First constexpr apply(First first, IntTypes... args) {
                return Layout::template find_val< MaxIndex, short_t, 1 >(first, args...);
            }
        };

        /**@brief functor to assign all the strides */
        template < int_t MaxIndex, typename Layout >
        struct assign_all_strides {

            template < int_t T >
            using lambda = next_stride< MaxIndex - T, MaxIndex, Layout >;

            template < typename... UIntType, typename Dummy = all_integers< UIntType... > >
            static constexpr array< int_t, MaxIndex > apply(UIntType... args) {
                using seq =
                    apply_gt_integer_sequence< typename make_gt_integer_sequence< int_t, sizeof...(args) >::type >;
                return seq::template apply< array< int_t, MaxIndex >, lambda >((int_t)args...);
            }
        };

#endif

        /**@brief struct to compute the total offset (the sum of the i,j,k indices times their respective strides)
 */
        template < ushort_t Id, typename Layout >
        struct compute_offset {
            static const ushort_t space_dimensions = Layout::length;

            /**interface with an array of coordinates as argument
               \param strides the strides
               \param indices the array of coordinates
            */ template < typename IntType, typename StridesVector >
            GT_FUNCTION static constexpr int_t apply(StridesVector const &RESTRICT strides_, IntType *indices_) {
                return strides_[space_dimensions - Id] *
                           Layout::template find_val< space_dimensions - Id, int, 0 >(indices_) +
                       compute_offset< Id - 1, Layout >::apply(strides_, indices_);
            }

#ifdef CXX11_ENABLED
            /**interface with the coordinates as variadic arguments
               \param strides the strides
               \param indices comma-separated list of coordinates
            */
            template < typename StridesVector, typename... Int >
            GT_FUNCTION static constexpr int_t apply(StridesVector const &RESTRICT strides_, Int const &... indices_) {
                return strides_[space_dimensions - Id] *
                           Layout::template find_val< space_dimensions - Id, int, 0 >(indices_...) +
                       compute_offset< Id - 1, Layout >::apply(strides_, indices_...);
            }
#endif
            /**interface with the coordinates as a tuple
               \param strides the strides
               \param indices tuple of coordinates
            */
            template < typename Tuple, typename StridesVector >
            GT_FUNCTION static constexpr int_t apply(StridesVector const &RESTRICT strides_, Tuple const &indices_) {
                return (int_t)strides_[space_dimensions - Id] *
                           Layout::template find_val< space_dimensions - Id, uint_t, 0 >(indices_) +
                       compute_offset< Id - 1, Layout >::apply(strides_, indices_);
            }
        };

        /**@brief stops the recursion
         */
        template < typename Layout >
        struct compute_offset< 1, Layout > {
            static const ushort_t space_dimensions = Layout::length;

            template < typename IntType, typename StridesVector >
            GT_FUNCTION static constexpr int_t apply(StridesVector const &RESTRICT /*strides*/, IntType *indices_) {
                return Layout::template find_val< space_dimensions - 1, int, 0 >(indices_);
            }

#ifdef CXX11_ENABLED
            template < typename StridesVector, typename... IntType >
            GT_FUNCTION static constexpr int_t apply(
                StridesVector const &RESTRICT /*strides*/, IntType const &... indices_) {
                return Layout::template find_val< space_dimensions - 1, int, 0 >(indices_...);
            }
#endif
            /**interface with the coordinates as a tuple
               \param strides the strides
               \param indices tuple of coordinates
            */
            template < typename Tuple, typename StridesVector >
            GT_FUNCTION static constexpr int_t apply(StridesVector const &RESTRICT /*strides*/, Tuple const &indices_) {
                return Layout::template find_val< space_dimensions - 1, int, 0 >(indices_);
            }
        };

        /**@brief metafunction to access a type sequence at a given position, numeration from 0

           he types in the sequence must define a 'super' type. Can be seen as a compile-time equivalent of a
           linked-list.
        */
        template < int_t ID, typename Sequence >
        struct access {
            BOOST_STATIC_ASSERT(ID > 0);
            // BOOST_STATIC_ASSERT(ID<=Sequence::n_fields);
            typedef typename access< ID - 1, typename Sequence::super >::type type;
        };

        /**@brief template specialization to stop the recursion*/
        template < typename Sequence >
        struct access< 0, Sequence > {
            typedef Sequence type;
        };

        /**@brief recursively advance the ODE finite difference for all the field dimensions*/
        template < short_t Dim >
        struct advance_recursive {
            template < typename This >
            GT_FUNCTION void apply(This *t) {
                t->template advance< Dim >();
                advance_recursive< Dim - 1 >::apply(t);
            }
        };

        /**@brief template specialization to stop the recursion*/
        template <>
        struct advance_recursive< 0 > {
            template < typename This >
            GT_FUNCTION void apply(This *t) {
                t->template advance< 0 >();
            }
        };

    } // namespace _impl

    template < enumtype::execution Execution >
    struct increment_policy;

    template <>
    struct increment_policy< enumtype::forward > {

        template < typename Lhs, typename Rhs >
        GT_FUNCTION static void apply(Lhs &lhs, Rhs const &rhs) {
            lhs += rhs;
        }
    };

    template <>
    struct increment_policy< enumtype::backward > {

        template < typename Lhs, typename Rhs >
        GT_FUNCTION static void apply(Lhs &lhs, Rhs const &rhs) {
            lhs -= rhs;
        }
    };

    namespace _debug {
#ifndef NDEBUG
        struct print_pointer {
            template < typename StorageType >
            GT_FUNCTION_WARNING void operator()(pointer<StorageType> s) const {
                printf("Pointer Value %x\n", s.get());
            }
        };
#endif
    } // namespace _debug
} // namesapace gridtools
