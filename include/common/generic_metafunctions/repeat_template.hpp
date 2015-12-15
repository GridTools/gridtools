#pragma once

namespace gridtools{

    template<typename UInt, UInt C, ushort_t N>
    struct expand_to_gt_integer_sequence : concat<typename expand_to_gt_integer_sequence<UInt, C, N/2>::type, typename expand_to_gt_integer_sequence<UInt, C, N - N/2>::type >::type{};

    template<typename UInt, UInt C> struct expand_to_gt_integer_sequence<UInt, C, 0> : gt_integer_sequence<UInt>{};
    template<typename UInt, UInt C> struct expand_to_gt_integer_sequence<UInt, C, 1> : gt_integer_sequence<UInt,C>{};

    template<typename Seq, template<ushort_t ...> class  Lambda>
    struct expand_recursively;

    template<template<ushort_t ...> class Lambda, ushort_t ... Ints>
    struct expand_recursively<gt_integer_sequence<ushort_t, Ints ...>, Lambda >{
        typedef Lambda<Ints ... > type;
    };

    template<typename Constant, typename Length, template<ushort_t ... T> class Lambda >
    struct repeat_template{
        typedef typename expand_recursively<typename expand_to_gt_integer_sequence<ushort_t, Constant::value, Length::value>::type, Lambda>::type type;
    };


    template<ushort_t Constant, ushort_t Length, template<ushort_t ... T> class Lambda >
    struct repeat_template_c{
        typedef typename expand_recursively<typename expand_to_gt_integer_sequence<ushort_t, Constant, Length>::type, Lambda>::type type;
    };
}//namespace gridtools
