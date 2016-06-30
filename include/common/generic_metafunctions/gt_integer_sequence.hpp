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

#include <functional>
#include <boost/proto/traits.hpp>
#include "common/defs.hpp"
#include "common/host_device.hpp"
#include "common/is_aggregate.hpp"

namespace gridtools {

#ifdef CXX11_ENABLED

    /**
       @brief helper struct to use an integer sequence in order to fill a generic container

       can be used with an arbitrary container with elements of the same type (not a tuple),
       it is consexpr constructable.
     */
    template < typename UInt, UInt... Indices >
    struct gt_integer_sequence {
        using type = gt_integer_sequence;
    };

    /** @bief concatenates two integer sequences*/
    template < class S1, class S2 >
    struct concat;

    template < typename UInt, UInt... I1, UInt... I2 >
    struct concat< gt_integer_sequence< UInt, I1... >, gt_integer_sequence< UInt, I2... > >
        : gt_integer_sequence< UInt, I1..., (sizeof...(I1) + I2)... > {};

    /** @brief constructs an integer sequence

        @tparam N number larger than 2, size of the integer sequence
     */
    template < typename UInt, uint_t N >
    struct make_gt_integer_sequence : concat< typename make_gt_integer_sequence< UInt, N / 2 >::type,
                                          typename make_gt_integer_sequence< UInt, N - N / 2 >::type >::type {};

    template < typename UInt >
    struct make_gt_integer_sequence< UInt, 0 > : gt_integer_sequence< UInt > {};
    template < typename UInt >
    struct make_gt_integer_sequence< UInt, 1 > : gt_integer_sequence< UInt, 0 > {};

    // with CXX14 the gt_integer_sequence from the standard can directly replace this one:
    // template <typename UInt, UInt ... Indices>
    // using gt_integer_sequence=std::integer_sequence<UInt, Indices ...>;

    // template<typename UInt, uint_t N>
    // using make_gt_integer_sequence=std::make_integer_sequence<UInt, N>;

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
           @brief returns a container constructed by applying a unary lambda function to each argument of the
           constructor
           The lambda applied is templated with an index which identifies the current argument. This allow
           to define specialised behaviour of the lambda for the specific arguments.

           \tparam Container the type of the container to be constructed
           \tparam Lambda the lambda template callable
           \tparam ExtraTypes the types of the input arguments to the lambda
         */
        template < typename Container,
            template < UInt T > class Lambda,
            typename... ExtraTypes,
            typename boost::disable_if< typename is_aggregate< Container >::type, int >::type = 0 >
        GT_FUNCTION static constexpr Container apply(ExtraTypes const &... args_) {
            return Container(Lambda< Indices >::apply(args_...)...);
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
           @brief duplicated interface for the case in which the container is an aggregator
         */
        template < typename Container,
            template < UInt T > class Lambda,
            typename... ExtraTypes,
            typename boost::enable_if< typename is_aggregate< Container >::type, int >::type = 0 >
        GT_FUNCTION static constexpr Container apply(ExtraTypes const &... args_) {
            return Container{Lambda< Indices >::apply(args_...)...};
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
            return Container(lambda(Indices, args_...)...);
        }
    };

#endif
} // namespace gridtools
