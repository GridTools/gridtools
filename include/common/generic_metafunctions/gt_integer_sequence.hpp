#pragma once

namespace gridtools{

#ifdef CXX11_ENABLED
    template<uint_t ... Indices> struct gt_integer_sequence{
        using type = gt_integer_sequence;

        /** @brief constructs and returns a Container initialized by Lambda<I>::apply(args_...)
            for all the indices I in the sequence*/
        template<typename Container, template <int_t T> class Lambda, typename ... UIntTypes>
        static constexpr Container apply(UIntTypes ... args_ ){
            return Container(Lambda<Indices>::apply(args_...) ...) ;
        }
    };

    /** @bief concatenates two integer sequences*/
    template<class S1, class S2> struct concat;

    template<uint_t... I1, uint_t... I2>
    struct concat<gt_integer_sequence<I1...>, gt_integer_sequence<I2...>>
        : gt_integer_sequence<I1..., (sizeof...(I1)+I2)...>{};

    /** @brief constructs an integer sequence

        @tparam N number larger than 2, size of the integer sequence
     */
    template<unsigned N>
    struct gt_make_integer_sequence : concat<typename gt_make_integer_sequence<N/2>::type, typename gt_make_integer_sequence<N - N/2>::type >::type{};

    template<> struct gt_make_integer_sequence<0> : gt_integer_sequence<>{};
    template<> struct gt_make_integer_sequence<1> : gt_integer_sequence<0>{};
#endif
} //namespace gridtools
