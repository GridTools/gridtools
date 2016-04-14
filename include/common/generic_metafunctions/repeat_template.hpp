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

        template < typename Seq, typename Value, template < Value... > class Lambda, Value... InitialValues >
        struct expand_recursively;

        template < typename Value, template < Value... > class Lambda, Value... Ints, Value... InitialValues >
        struct expand_recursively< gt_integer_sequence< Value, Ints... >, Value, Lambda, InitialValues... > {
            typedef Lambda< InitialValues..., Ints... > type;
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
       Optionally a set of initial values to start filling the template class can be passed
     */
    template < typename Constant, typename Length, typename Value, template < Value... T > class Lambda, Value... InitialValues >
    struct repeat_template {
        typedef typename _impl::expand_recursively<
            typename _impl::expand_to_gt_integer_sequence< Value, Constant::value, Length::value >::type,
            Value,
            Lambda,
            InitialValues... >::type type;
    };

    /**
       @brief Metafunction for creating a template class with an arbitrary length template parameter pack.

       Usage example:
       I have a class template halo< .... >, and I want to fill it by repeating N times the same number H
       \verbatim
       repeat_template_c<H, N, halo>
       \endverbatim
       Optionally a set of initial values to start filling the template class can be passed
    */
    template < ushort_t Constant, ushort_t Length, typename Value, template < Value... T > class Lambda, Value... InitialValues >
    struct repeat_template_c {
        typedef typename _impl::expand_recursively<
            typename _impl::expand_to_gt_integer_sequence< Value, Constant, Length >::type,
            Value,
            Lambda,
            InitialValues... >::type type;
    };

} // namespace gridtools
