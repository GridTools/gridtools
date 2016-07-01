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

#include "./mss.hpp"
#include "./reductions/reduction_descriptor.hpp"

/**
@file
@brief Implementation of metafunctions to linearize the multi stage stencil descriptors.
Linearizing means to unwrap the independent_esfs.
*/
namespace gridtools {
    namespace _impl {
        /*
          This metafunction traverses an array of esfs that may contain indendent_esfs.

          This is used by mss_descriptor_linear_esf_sequence and
          sequence_of_is_independent_esf.  The first one creates a
          vector of esfs the second one a vector of boolean types that
          are true if an esf is in an independent section, and false
          otherwise.

          To reuse the code, the values to push are passed as template
          arguments. By passing a placeholder as value to push, then
          the corresponding folds will "substitute" that, thus
          allowing to process sequence elements. Otherwise boolean
          types are passed and those will be pushed into the result
          vector. (This is a rework of Paolo's code that was failing
          for nested independents.
         */
        template < typename EsfsVector, typename PushRegular, typename PushIndependent = PushRegular >
        struct linearize_esf_array {

            GRIDTOOLS_STATIC_ASSERT((is_sequence_of< EsfsVector, is_esf_descriptor >::value), "Wrong type");

            template < typename Vector, typename Element >
            struct push_into {
                typedef typename boost::mpl::push_back< Vector, Element >::type type;
            };

            template < typename Vector, typename Independents >
            struct push_into< Vector, independent_esf< Independents > > {
                typedef typename boost::mpl::fold< Independents,
                    Vector,
                    push_into< boost::mpl::_1, PushIndependent > >::type type;
            };

            typedef typename boost::mpl::fold< EsfsVector,
                boost::mpl::vector0<>,
                push_into< boost::mpl::_1, PushRegular > >::type type;
        };
    } // namespace _impl

    /**
       @brief constructs an mpl vector of esf, linearizig the mss tree.

       Looping over all the esfs at compile time.
       if found independent esfs, they are also included in the linearized vector with a nested fold.

       NOTE: the nested make_independent calls get also linearized
     */
    template < typename T >
    struct mss_descriptor_linear_esf_sequence;

    template < typename ExecutionEngine, typename EsfDescrSequence, typename CacheSequence >
    struct mss_descriptor_linear_esf_sequence< mss_descriptor< ExecutionEngine, EsfDescrSequence, CacheSequence > > {
        typedef typename _impl::linearize_esf_array< EsfDescrSequence, boost::mpl::_2 >::type type;
    };

    template < typename ReductionType, typename BinOp, typename EsfDescrSequence >
    struct mss_descriptor_linear_esf_sequence< reduction_descriptor< ReductionType, BinOp, EsfDescrSequence > > {
        typedef typename _impl::linearize_esf_array< EsfDescrSequence, boost::mpl::_2 >::type type;
    };

    /**
       @brief pushes an element in a vector based on the fact that an ESF is independent or not

       Helper metafunction, used by other metafunctions
    */
    template < typename State, typename SubArray, typename VectorComponent >
    struct keep_scanning_lambda
        : boost::mpl::fold< typename SubArray::esf_list,
              State,
              boost::mpl::if_< is_independent< boost::mpl::_2 >,
                                keep_scanning_lambda< boost::mpl::_1, boost::mpl::_2, VectorComponent >,
                                boost::mpl::push_back< boost::mpl::_1, VectorComponent > > > {};

    /**
       @brief linearizes the ESF tree and returns a vector

       Helper metafunction, used by other metafunctions
    */
    template < typename Array, typename Argument, template < typename, typename > class KeepScanning >
    struct linearize_esf_array_lambda : boost::mpl::fold< Array,
                                            boost::mpl::vector0<>,
                                            boost::mpl::if_< is_independent< boost::mpl::_2 >,
                                                              KeepScanning< boost::mpl::_1, boost::mpl::_2 >,
                                                              boost::mpl::push_back< boost::mpl::_1, Argument > > > {};

    /**
       @brief constructs an mpl vector of booleans, linearizing the mss tree and attachnig a true or false flag
       depending wether the esf is independent or not

       the code is very similar as in the metafunction above
    */
    template < typename T >
    struct sequence_of_is_independent_esf;

    template < typename ExecutionEngine, typename EsfDescrSequence, typename CacheSequence >
    struct sequence_of_is_independent_esf< mss_descriptor< ExecutionEngine, EsfDescrSequence, CacheSequence > > {

        template < typename State, typename SubArray >
        struct keep_scanning : keep_scanning_lambda< State, SubArray, boost::mpl::true_ > {};

        template < typename Array >
        struct linearize_esf_array : linearize_esf_array_lambda< Array, boost::mpl::false_, keep_scanning > {};

        typedef typename linearize_esf_array< EsfDescrSequence >::type type;
    };

    // /**
    //    @brief constructs an mpl vector of booleans, linearizing the mss tree and attachnig a true or false flag
    //    depending wether the esf is independent or not

    //    the code is very similar as in the metafunction above
    //  */
    // template < typename T >
    // struct sequence_of_is_independent_esf;

    // template < typename ExecutionEngine, typename EsfDescrSequence, typename CacheSequence >
    // struct sequence_of_is_independent_esf< mss_descriptor< ExecutionEngine, EsfDescrSequence, CacheSequence > > {
    //     typedef typename _impl::linearize_esf_array< EsfDescrSequence, boost::false_type, boost::true_type >::type
    //     type;
    // };

    template < typename ReductionType, typename BinOp, typename EsfDescrSequence >
    struct sequence_of_is_independent_esf< reduction_descriptor< ReductionType, BinOp, EsfDescrSequence > > {
        typedef typename _impl::linearize_esf_array< EsfDescrSequence, boost::false_type, boost::true_type >::type type;
    };

} // namespace gridtools
