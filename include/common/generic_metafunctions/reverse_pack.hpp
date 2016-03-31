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
