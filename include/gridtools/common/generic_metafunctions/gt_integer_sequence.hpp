/*
  GridTools Libraries

  Copyright (c) 2017, ETH Zurich and MeteoSwiss
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  1. Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

  For information: http://eth-cscs.github.io/gridtools/
*/
#pragma once

#include <functional>
#include <boost/proto/traits.hpp>
#include "common/defs.hpp"
#include "common/host_device.hpp"

namespace gridtools {

    /** \ingroup common
        @{
        \ingroup allmeta
        @{
        \defgroup gtintegersequence GridTools Integer Sequence
        @{
    */

    /**
       @brief helper struct to use an integer sequence in order to fill a generic container

       can be used with an arbitrary container with elements of the same type (not a tuple),
       it is consexpr constructable.
     */
    template < typename UInt, UInt... Indices >
    struct gt_integer_sequence {
        using type = gt_integer_sequence;
        using value_type = UInt;
        static constexpr std::size_t size() noexcept { return sizeof...(Indices); }
    };

    /**
       @brief Given two integer sequences returns the sequence with
       the two sequences it returns a sequence in which the second
       sequence is attached to the first one (without any
       transformation, as it the case of gridtools::concat)

       \tparam S1 Frist sequence
       \tparam S2 Second sequence

     */
    template < typename S1, typename S2 >
    struct append;

    template < typename Uint, Uint... S1, Uint... S2 >
    struct append< gt_integer_sequence< Uint, S1... >, gt_integer_sequence< Uint, S2... > > {
        using type = gt_integer_sequence< Uint, S1..., S2... >;
    };

    /** @bief concatenates two integer sequences*/
    namespace _impl {
        template < typename Seq, size_t Size, size_t Rem >
        struct expand_integer_sequence;

        template < typename UInt, UInt... Is, size_t Size >
        struct expand_integer_sequence< gt_integer_sequence< UInt, Is... >, Size, 0 > {
            using type = gt_integer_sequence< UInt, Is..., (Size + Is)... >;
        };

        template < typename UInt, UInt... Is, size_t Size >
        struct expand_integer_sequence< gt_integer_sequence< UInt, Is... >, Size, 1 > {
            using type = gt_integer_sequence< UInt, Is..., (Size + Is)..., 2 * Size >;
        };

        template < typename UInt, size_t N >
        struct generate_integer_sequence {
            using type = typename expand_integer_sequence< typename generate_integer_sequence< UInt, N / 2 >::type,
                N / 2,
                N % 2 >::type;
        };

        template < typename UInt >
        struct generate_integer_sequence< UInt, 0 > {
            using type = gt_integer_sequence< UInt >;
        };
    }

    /** @brief constructs an integer sequence

        @tparam N number larger than 2, size of the integer sequence
     */
    template < typename UInt, UInt N >
    using make_gt_integer_sequence = typename _impl::generate_integer_sequence< UInt, N >::type;

    template < size_t... Ints >
    using gt_index_sequence = gt_integer_sequence< size_t, Ints... >;

    template < size_t N >
    using make_gt_index_sequence = make_gt_integer_sequence< size_t, N >;

    template < class... Ts >
    using gt_index_sequence_for = make_gt_index_sequence< sizeof...(Ts) >;

    // with CXX14 the gt_integer_sequence from the standard can directly replace this one:
    // template <typename UInt, UInt ... Indices>
    // using gt_integer_sequence=std::integer_sequence<UInt, Indices ...>;

    // template<typename UInt, UInt N>
    // using make_gt_integer_sequence=std::make_integer_sequence<UInt, N>;

    namespace impl {
        template < typename... U >
        GT_FUNCTION constexpr uint_t void_lambda(U... args) {
            return 0;
        }
    }

    /**
       @brief generic definition (never instantiated)
     */
    template < typename UInt >
    struct apply_gt_integer_sequence {
        template < typename Container, template < UInt T > class Lambda, typename... ExtraTypes >
        GT_FUNCTION static constexpr Container apply(ExtraTypes const &... args_) {
            GRIDTOOLS_STATIC_ASSERT((boost::is_same< Container, Container >::value),
                "ERROR: apply_gt_integer_sequence only accepts a gt_integer_sequence type. Check the call");
            return Container(args_...);
        }
    };

    /** @brief constructs and returns a Container initialized by Lambda<I>::apply(args_...)
        for all the indices I in the sequence

        @tparam Container is the container to be filled
        @tparam Lambda is a metafunction templated with an integer, whose static member
        function "apply" returns an element of the container
        @tparam ExtraTypes are the types of the arguments to the method "apply" (deduced by the compiler)

        The type of the Container members must correspond to the return types of the apply method in
        the user-defined Lambda functor.
    */
    template < typename UInt, UInt... Indices >
    struct apply_gt_integer_sequence< gt_integer_sequence< UInt, Indices... > > {

        /**
           @brief returns a container type constructed with template arguments which are the result of
         *  applying a unary lambda function to each template argument of the ExtraTypes
         *  The lambda applied is templated with an index which identifies the current argument. This allow
         *  to define specialised behaviour of the lambda for the specific arguments.
         *
         *  Container's value type is currently restricted to int_t, due to limited compiler support of
         *  template template parameters that depend on a previous template argument.
         *
         *  \tparam Int type of the template parameters that the Container accepts
         *  \tparam Container the type of the container to be constructed
         *  \tparam Lambda the lambda template callable
         *  \tparam ExtraTypes the types of the input arguments to the lambda
         */
        template < template < int_t... t > class Container,
            template < UInt TT, typename > class Lambda,
            typename... ExtraTypes >
        struct apply_t {
            using type = Container< Lambda< Indices, ExtraTypes >::value... >;
        };

        /**
           @brief returns a container constructed by applying a unary lambda function to each argument of the
           constructor
           The lambda applied is templated with an index which identifies the current argument. This allow
           to define specialised behaviour of the lambda for the specific arguments.

           \tparam Container the type of the container to be constructed
           \tparam Lambda the lambda template callable
           \tparam ExtraTypes the types of the input arguments to the lambda
         */
        template < typename Container, template < UInt T > class Lambda, typename... ExtraTypes >
        GT_FUNCTION static constexpr Container apply(ExtraTypes const &... args_) {
            return Container{Lambda< Indices >::apply(args_...)...};
        }

        /**
           @brief applies a lambda function to the transformed argument pack.
           The original argument pack provided by the user args_... is transformed by the apply method of the
           MetaFunctor
           provided. The resulting argument pack is used to call the lambda.

           The metafunctor applied is templated with an index which identifies the current argument. This allow
           to define specialised behaviour of the functor for the specific arguments.

           \tparam Lambda lambda function applied to the transformed argument pack
           \tparam MetaFunctor functor that is transforming each of the arguments of the variadic pack
           \tparam AdditionalArg additional argument passed to the lambda at the end of the pack
           \tparam ExtraTypes variadic pack of arguments to be passed to the lambda
         */
        template < typename ReturnType,
            typename Lambda,
            template < UInt T > class MetaFunctor,
            typename AdditionalArg,
            typename... ExtraTypes >
        GT_FUNCTION static constexpr ReturnType apply_lambda(
            Lambda lambda, AdditionalArg add_arg, ExtraTypes const &... args_) {
            return lambda(MetaFunctor< Indices >::apply(args_...)..., add_arg);
        }

        /**
           @brief applies a lambda function to the transformed argument pack.
           The lambda function does not return any value, like in other specializations of this class, but
           simply apply a void action of each index of the integer sequence.
           This functionality is equivalent to a boost::mpl::for_each

           \tparam Lambda lambda function applied to the transformed argument pack
           \tparam MetaFunctor functor that is transforming each of the arguments of the variadic pack
           \tparam ExtraTypes variadic pack of arguments to be passed to the lambda
         */
        template < template < UInt T > class MetaFunctor, typename... ExtraTypes >
        GT_FUNCTION static constexpr uint_t apply_void_lambda(ExtraTypes &... args_) {
            return impl::void_lambda(MetaFunctor< Indices >::apply(args_...)...);
        }

        /**
          @brief applies a unary lambda to a sequence of arguments, and returns a container constructed
          using such lambda

          \tparam Container the container type
          \tparam Lambda the callable lambda type
          \tparam ExtraArgs the arguments types
          \param arg_ the input values, i.e. a variadic sequence.
        */
        template < typename Container, template < UInt T > class Lambda, typename... ExtraTypes >
        GT_FUNCTION static constexpr Container apply_zipped(ExtraTypes const &... arg_) {
            return Container(Lambda< Indices >::apply(arg_)...);
        }

        /**
           @brief applies a templated lambda metafunction to generate
           a new type which is 'zipping' the integer sequence indices and the input indices

           \tparam Container the output type
           \tparam Lambda the lambda metafunction, mapping the integer sequence indices to the input indices
           \tparam ExtraTypes input types
         */
        template < template < typename... U > class Container,
            template < UInt TT, UInt UU > class Lambda,
            UInt... ExtraTypes >
        struct apply_tt {
            using type = Container< Lambda< Indices, ExtraTypes >... >;
        };

        /**
           @brief same as before, but with non-static lambda taking as first argument the index
        */
        template < typename Container, class Lambda, typename... ExtraTypes >
        GT_FUNCTION static constexpr Container apply(Lambda lambda, ExtraTypes &... args_) {
            return Container{lambda(Indices, args_...)...};
        }
    };
    /** @} */
    /** @} */
    /** @} */

} // namespace gridtools
