#pragma once

namespace gridtools{

#ifdef CXX11_ENABLED
    template<typename UInt, UInt ... Indices> struct gt_integer_sequence{
        using type = gt_integer_sequence;

        /** @brief constructs and returns a Container initialized by Lambda<I>::apply(args_...)
            for all the indices I in the sequence*/
        template<typename Container, template <UInt T> class Lambda, typename ... Arguments>
        GT_FUNCTION
        static constexpr Container apply(Arguments ... args_ ){
            return Container{Lambda<Indices>::apply(args_...) ...} ;
        }
    };

    /** @bief concatenates two integer sequences*/
    template<class S1, class S2> struct concat;

    template<typename UInt, UInt... I1, UInt... I2>
    struct concat<gt_integer_sequence<UInt, I1...>, gt_integer_sequence<UInt, I2...>>
        : gt_integer_sequence<UInt, I1..., (sizeof...(I1)+I2)...>{};

    /** @brief constructs an integer sequence

        @tparam N number larger than 2, size of the integer sequence
     */
    template<typename UInt, uint_t N>
    struct gt_make_integer_sequence : concat<typename gt_make_integer_sequence<UInt, N/2>::type, typename gt_make_integer_sequence<UInt, N - N/2>::type >::type{};

    template<typename UInt> struct gt_make_integer_sequence<UInt, 0> : gt_integer_sequence<UInt>{};
    template<typename UInt> struct gt_make_integer_sequence<UInt, 1> : gt_integer_sequence<UInt,0>{};
#endif
} //namespace gridtools
