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
namespace gridtools {
/**@brief metafunction for applying a parameter pack in reversed order

   usage example:
   reverse<4, 3, 2>::apply<ToBeReversed, ExtraArgument, 8>::type::type
   gives
   ToBeReversed<ExtraArgument, 8, 2, 3, 4>::type
 */
// forward decl
#ifdef CXX11_ENABLED
    template < uint_t... Tn >
    struct reverse_pack;

    // recursion anchor
    template <>
    struct reverse_pack<> {
        template < template < typename, uint_t... > class ToBeReversed, typename ExtraArgument, uint_t... Un >
        struct apply {
            typedef ToBeReversed< ExtraArgument, Un... > type;
        };
    };

    // recursion
    template < uint_t T, uint_t... Tn >
    struct reverse_pack< T, Tn... > {
        template < template < typename ExtraArgument, uint_t... > class ToBeReversed,
            typename ExtraArgument,
            uint_t... Un >
        struct apply {
            // bubble 1st parameter backwards
            typedef typename reverse_pack< Tn... >::template apply< ToBeReversed, ExtraArgument, T, Un... >::type type;
        };
    };
#endif
} // namespace gridtools
