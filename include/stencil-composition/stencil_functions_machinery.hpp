#pragma once

#include "../common/generic_metafunctions/variadic_to_vector.hpp"
#include <boost/fusion/include/as_vector.hpp>
#include "accessor.hpp"

namespace gridtools {


    /** Metafunction to compute the index of the first accessor in the
        list of accessors to be written.
     */
    template <typename Functor>
    struct _get_index_of_first_non_const {

        template <int I, int L, typename List>
        struct scan_for_index {
            using type = typename boost::mpl::if_
                <typename std::is_const<typename boost::mpl::at_c<List, I>::type >::type,
                 typename scan_for_index<I+1, L, List>::type,
                 std::integral_constant<int, I>
                 >::type;
        };

        template <int I, typename List>
        struct scan_for_index<I, I, List> {
            using type = static_int<-1>;
        };

        static const int value = scan_for_index
            <0,
             boost::mpl::size<typename Functor::arg_list>::value,
             typename Functor::arg_list>::type::value;
    };

    /** Metafunction to check that there is only one
        written argument in the argument list of a stencil
        operator, so that it is legal to call it as a
        function.
     */
    template <typename Functor>
    struct can_be_a_function {

        template <typename CurrentCount, typename CurrentArg>
        struct count_if_written {
            typedef typename boost::mpl::if_
                <typename std::is_const<CurrentArg>::type,
                 CurrentCount,
                 static_int<CurrentCount::value+1>
                 >::type type;
        };

        typedef typename boost::mpl::fold
            <typename Functor::arg_list,
             static_int<0>,
             count_if_written<boost::mpl::_1, boost::mpl::_2>
             >::type type;

        static const bool value = type::value==1;
    };


    /**
       In the context of stencil_functions, this type represents the
       aggregator/domain/evaluator to be passed to a stencil function,
       called within a stencil operator or another stencil function.
       Construct a new aggregator/domain from the initial one (which is
       usually an iterate_domain).  The indices in the seauence
       UsedIndices indicate which elements of the CallerAggregator are
       passed to the function. One of the indices correspond to the output
       argument which is a scalar and requires special attention.
    */
    template <typename CallerAggregator, int Offi, int Offj, int Offk, typename PassedAccessors, typename ReturnType, int OutArg>
    struct function_aggregator {
        CallerAggregator const& m_caller_aggregator;
        ReturnType __restrict__ * m_result;

        GT_FUNCTION
        function_aggregator(CallerAggregator const& caller_aggregator, ReturnType & result)
            : m_caller_aggregator(caller_aggregator)
            , m_result(&result)
        {}

        template <typename Accessor>
        GT_FUNCTION
        constexpr
        typename boost::enable_if_c<(Accessor::index_type::value < OutArg), ReturnType>::type const&
        operator()(Accessor const& accessor) const {
            return m_caller_aggregator
                (typename boost::mpl::at_c<PassedAccessors, Accessor::index_type::value>::type
                 (accessor.template get<2>()+Offi,
                  accessor.template get<1>()+Offj,
                  accessor.template get<0>()+Offk));
        }

        template <typename Accessor>
        GT_FUNCTION
        constexpr
        typename boost::enable_if_c<(Accessor::index_type::value > OutArg), ReturnType>::type const&
        operator()(Accessor const& accessor) const {
            return m_caller_aggregator
                (typename boost::mpl::at_c<PassedAccessors, Accessor::index_type::value-1>::type
                 (accessor.template get<2>()+Offi,
                  accessor.template get<1>()+Offj,
                  accessor.template get<0>()+Offk));
        }

        template <typename Accessor>
        GT_FUNCTION
        constexpr
        typename boost::enable_if_c<(Accessor::index_type::value == OutArg), ReturnType>::type&
        operator()(Accessor const&) const {
            // std::cout << "Giving the ref (OutArg=" << OutArg << ") " << m_result << std::endl;
            return *m_result;
        }

    };



    /** Main interface for calling stencil operators as functions.

        Usage C++11: call<functor, region>::[at<offseti, offsetj, offsetk>::]with(eval, accessors...);

        Usage : call<functor, region>::[at<offseti, offsetj, offsetk>::type::]with(eval, accessors...);
    */
    template <typename Functor, typename Region, int Offi=0, int Offj=0, int Offk=0>
    struct call {
        template <int I, int J, int K>
        struct at_ {
            typedef call<Functor, Region, I, J, K> type;
        };

        template <int I, int J, int K>
        using at = call<Functor, Region, I, J, K>;

        template <typename Eval, typename Funct>
        struct get_result_type {
            typedef accessor<_get_index_of_first_non_const<Funct>::value> accessor_t;

            typedef typename Eval::template accessor_return_type
                <accessor_t>::type tt;
            
            typedef typename tt::value_type type;
        };

        template <typename Evaluator, typename ...Args>
        GT_FUNCTION
        static
        typename get_result_type<Evaluator, Functor>::type
        with(Evaluator const& eval, Args const& ...args) {

            // TODO: Mauro: The following check does not seem to work. Waiting for next version
            // accessors to fix this
            static_assert(can_be_a_function<Functor>::value,
                          "Trying to invoke stencil operator with more than one output as a function\n");

            typedef typename get_result_type<Evaluator, Functor>::type result_type;

            result_type result;

            Functor::Do(function_aggregator
                <Evaluator,
                 Offi, Offj, Offk,
                 typename gridtools::variadic_to_vector<Args...>::type,
                 result_type,
                 _get_index_of_first_non_const<Functor>::value>(eval, result), Region());

            return result;
        }
    };

    /****************************************************************************************
     ****************************************************************************************
     ****************************************************************************************
     ****************************************************************************************
     ****************************************************************************************
     ****************************************************************************************
     ****************************************************************************************/



    // template <typename CallerAggregator, int Offi, int Offj, int Offk, typename PassedAccessors, typename ReturnType, int OutArg>
    // struct function_aggregator_offsets {
    //     typedef typename boost::fusion::result_of::as_vector<PassedAccessors>::type accessors_list_t;
    //     CallerAggregator const& m_caller_aggregator;
    //     ReturnType mutable m_result;
    //     accessors_list_t const& m_accessors_list;

    //     GT_FUNCTION
    //     function_aggregator_offsets(CallerAggregator const& caller_aggregator, accessors_list_t const& list )
    //         : m_caller_aggregator(caller_aggregator)
    //         , m_result(0.0)
    //         , m_accessors_list(list)
    //     {}

    //     GT_FUNCTION
    //     ReturnType result() const { return m_result; }

    //     template <typename Accessor>
    //     GT_FUNCTION
    //     constexpr
    //     typename boost::enable_if_c<(Accessor::index_type::value < OutArg), ReturnType>::type
    //     operator()(Accessor const& accessor) const {
    //         // std::cout << accessor.template get<2>()
    //         //           << ", " << accessor.template get<1>()
    //         //           << ", " << accessor.template get<0>()
    //         //           << std::endl;
    //         return m_caller_aggregator
    //             (typename boost::mpl::at_c<PassedAccessors, Accessor::index_type::value>::type
    //              (accessor.template get<2>()+Offi+boost::fusion::at_c<Accessor::index_type::value>(m_accessors_list).template get<2>(),
    //               accessor.template get<1>()+Offj+boost::fusion::at_c<Accessor::index_type::value>(m_accessors_list).template get<1>(),
    //               accessor.template get<0>()+Offk)+boost::fusion::at_c<Accessor::index_type::value>(m_accessors_list).template get<0>());
    //     }

    //     template <typename Accessor>
    //     GT_FUNCTION
    //     constexpr
    //     typename boost::enable_if_c<(Accessor::index_type::value > OutArg), ReturnType>::type
    //     operator()(Accessor const& accessor) const {
    //         // std::cout << accessor.template get<2>()
    //         //           << ", " << accessor.template get<1>()
    //         //           << ", " << accessor.template get<0>()
    //         //           << "  ---> ";
    //         // std::cout << m_caller_aggregator
    //         //     (typename boost::mpl::at_c<PassedAccessors, Accessor::index_type::value-1>::type(accessor.template get<2>()+Offi, accessor.template get<1>()+Offj, accessor.template get<0>()+Offk))
    //         //           << std::endl;
    //         return m_caller_aggregator
    //             (typename boost::mpl::at_c<PassedAccessors, Accessor::index_type::value-1>::type
    //              (accessor.template get<2>()+Offi+boost::fusion::at_c<Accessor::index_type::value-1>(m_accessors_list).template get<2>(),
    //               accessor.template get<1>()+Offj+boost::fusion::at_c<Accessor::index_type::value-1>(m_accessors_list).template get<1>(),
    //               accessor.template get<0>()+Offk+boost::fusion::at_c<Accessor::index_type::value-1>(m_accessors_list).template get<0>()));
    //     }

    //     template <typename Accessor>
    //     GT_FUNCTION
    //     constexpr
    //     typename boost::enable_if_c<(Accessor::index_type::value == OutArg), ReturnType>::type&
    //     operator()(Accessor const&) const {
    //         // std::cout << "Giving the ref (OutArg=" << OutArg << ") " << m_result << std::endl;
    //         return m_result;
    //     }

    // };


    // template <typename Evaluator, typename ResultType, typename Functor, int OutArg, int Offi, int Offj, int Offk>
    // struct insert_argument_offsets {
    //     Evaluator const& eval;

    //     GT_FUNCTION
    //     insert_argument_offsets(Evaluator const& eval)
    //         : eval(eval)
    //     {}

    //     template <typename ...Args>
    //     constexpr
    //     GT_FUNCTION
    //     function_aggregator_offsets<Evaluator, Offi, Offj, Offk, typename gridtools::variadic_to_vector<Args...>::type, ResultType, OutArg>
    //     operator()(Args const... args) const &
    //     {
    //         typedef function_aggregator_offsets<Evaluator, Offi, Offj, Offk, typename gridtools::variadic_to_vector<Args...>::type, ResultType, OutArg> f_aggregator_t;

    //         return f_aggregator_t(eval, typename f_aggregator_t::accessors_list_t(args...));
    //     }
    // };

    // template <typename Evaluator, typename Functor, typename Region, int OutArg, typename ResultType, int Offi, int Offj, int Offk>
    // struct call_the_damn_thing_offsets {
    //     Evaluator const& eval;

    //     GT_FUNCTION
    //     call_the_damn_thing_offsets(Evaluator const& eval)
    //         : eval(eval)
    //     {}

    //     template <typename ...Args>
    //     GT_FUNCTION
    //     ResultType then_do_it_damn_it(Args const& ...args) const {
    //         auto newargs = insert_argument_offsets<
    //             Evaluator,
    //             ResultType,
    //             Functor,
    //             _get_index_of_first_non_const<Functor>::type,
    //             Offi, Offj, Offk >::(eval, args...);
    //         Functor::Do(newargs, Region());
    //         return newargs.result();
    //     }
    // };

    // template <typename Functor, typename ResultType, typename Region, int Offi=0, int Offj=0, int Offk=0>
    // struct call_offsets {

    //     template <int I, int J, int K>
    //     struct at_ {
    //         typedef call<Functor, ResultType, Region, I, J, K> type;
    //     };

    //     template <int I, int J, int K>
    //     using at = call_offsets<Functor, ResultType, Region, I, J, K>;

    //     template <typename Evaluator, typename ...Args>
    //     GT_FUNCTION
    //     static ResultType with(Evaluator const& eval, Args const& ...args) {

    //         // Here several checks can be performed

    //         return call_the_damn_thing_offsets<
    //             Evaluator,
    //             Functor,
    //             Region,
    //             _get_index_of_first_non_const<Functor>::value,
    //             ResultType, Offi, Offj, Offk>(eval)
    //             .then_do_it_damn_it(args...);
    //     }
    // };

} // namespace gridtools
