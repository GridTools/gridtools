#pragma once
/**
   @file

   Metafunction for creating a template class with an arbitrary length template parameter pack.
   NOTE: gt_integer_sequence can be replaced by std::integer_sequence for C++14 and beyond.
*/

#include "gt_integer_sequence.hpp"

namespace gridtools {

    namespace _impl {
        template < typename T1, typename T2 >
        struct expand;

        template < typename UInt, UInt... I1, UInt... I2 >
        struct expand< gt_integer_sequence< UInt, I1... >, gt_integer_sequence< UInt, I2... > >
            : gt_integer_sequence< UInt, I1..., I2... > {};

        template < typename UInt, UInt C, ushort_t N >
        struct expand_to_gt_integer_sequence
            : expand< typename expand_to_gt_integer_sequence< UInt, C, N / 2 >::type,
                  typename expand_to_gt_integer_sequence< UInt, C, N - N / 2 >::type >::type {};

        template < typename UInt, UInt C >
        struct expand_to_gt_integer_sequence< UInt, C, 0 > : gt_integer_sequence< UInt > {};
        template < typename UInt, UInt C >
        struct expand_to_gt_integer_sequence< UInt, C, 1 > : gt_integer_sequence< UInt, C > {};

        template < typename Seq, template < ushort_t... > class Lambda >
        struct expand_recursively;

        template < template < ushort_t... > class Lambda, ushort_t... Ints >
        struct expand_recursively< gt_integer_sequence< ushort_t, Ints... >, Lambda > {
            typedef Lambda< Ints... > type;
        };
    }
    /**
       @brief Metafunction for creating a template class with an arbitrary length template parameter pack.

       Usage example:
       I have a class template halo< .... >, and I want to fill it by repeating N times the same number H (represented
       as an MPL type)
       \verbatim
       repeat_template<static_short<H>, static_short<N>, halo>
       \endverbatim
     */
    template < typename Constant, typename Length, template < ushort_t... T > class Lambda >
    struct repeat_template {
        typedef typename _impl::expand_recursively<
            typename _impl::expand_to_gt_integer_sequence< ushort_t, Constant::value, Length::value >::type,
            Lambda >::type type;
    };

    /**
       @brief Metafunction for creating a template class with an arbitrary length template parameter pack.

       Usage example:
       I have a class template halo< .... >, and I want to fill it by repeating N times the same number H
       \verbatim
       repeat_template_c<H, N, halo>
       \endverbatim
    */
    template < ushort_t Constant, ushort_t Length, template < ushort_t... T > class Lambda >
    struct repeat_template_c {
        typedef typename _impl::expand_recursively<
            typename _impl::expand_to_gt_integer_sequence< ushort_t, Constant, Length >::type,
            Lambda >::type type;
    };
} // namespace gridtools
