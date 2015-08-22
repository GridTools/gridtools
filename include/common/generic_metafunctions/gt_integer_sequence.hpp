#pragma once

#include <functional>

namespace gridtools{

#ifdef CXX11_ENABLED

    /**
       @brief helper struct to use an integer sequence in order to fill a generic container

       can be used with an arbitrary container with elements of the same type (not a tuple),
       it is consexpr constructable.
     */
    template<uint_t ... Indices> struct gt_integer_sequence{
        using type = gt_integer_sequence;

        /** @brief constructs and returns a Container initialized by Lambda<I>::apply(args_...)
            for all the indices I in the sequence

            @tparam Container is the container to be filled
            @tparam Lambda is a metafunction templated with an integer, whose static member
            function "apply" returns an element of the container
            @tparam ExtraTypes are the types of the arguments to the method "apply" (deduced by the compiler)

            The type of the Container members must correspond to the return types of the apply method in
            the user-defined Lambda functor.
        */
        template<typename Container, template <int_t T> class Lambda, typename ... ExtraTypes>
        static constexpr Container apply(ExtraTypes& ... args_ ){
            return Container(Lambda<Indices>::apply(args_...) ...) ;
        }

        /**
           @brief same as before, but with non-static lambda taking as first argument the index
         */
        template<typename Container, class Lambda, typename ... ExtraTypes>
            static constexpr Container apply(Lambda lambda, ExtraTypes& ... args_ ){
            return Container(lambda(Indices, args_...) ...) ;
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
