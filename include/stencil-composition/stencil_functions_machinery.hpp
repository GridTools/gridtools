#pragma once

#include "../common/generic_metafunctions/variadic_to_vector.hpp"

namespace gridtools {

    template <typename T>
    struct wrap {
        T x;

        wrap(T x) : x(x) {}

        wrap& operator=(T b) {
            std::cout << "Assign! " << b << std::endl;
            x = b;
            return *this;
        }

        operator double() const {std::cout << "Converting" << std::endl; return x;}
    };

    template <typename T>
    std::ostream& operator<<(std::ostream& s, wrap<T> const& a) {
        return s << a.x;
    }

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
    template <typename CallerAggregator, typename PassedAccessors, typename ReturnType, int OutArg>
    struct function_aggregator {
        CallerAggregator const& m_caller_aggregator;
        ReturnType mutable m_result;

        function_aggregator(CallerAggregator const& caller_aggregator)
            : m_caller_aggregator(caller_aggregator)
            , m_result(0.0)
        {}

        ReturnType result() const { return m_result; }

        template <typename Accessor>
        typename boost::enable_if_c<(Accessor::index_type::value < OutArg), ReturnType>::type
        operator()(Accessor const& accessor) const {
            std::cout << accessor.template get<2>()
                      << ", " << accessor.template get<1>()
                      << ", " << accessor.template get<0>()
                      << std::endl;
            return m_caller_aggregator
                (typename boost::mpl::at_c<PassedAccessors, Accessor::index_type::value>::type
                 (accessor.template get<2>(), accessor.template get<1>(), accessor.template get<0>()));
        }

        template <typename Accessor>
        typename boost::enable_if_c<(Accessor::index_type::value > OutArg), ReturnType>::type
        operator()(Accessor const& accessor) const {
            std::cout << accessor.template get<2>()
                      << ", " << accessor.template get<1>()
                      << ", " << accessor.template get<0>()
                      << "  ---> ";
            std::cout << m_caller_aggregator
                (typename boost::mpl::at_c<PassedAccessors, Accessor::index_type::value-1>::type(accessor.template get<2>(), accessor.template get<1>(), accessor.template get<0>()))
                      << std::endl;
            return m_caller_aggregator
                (typename boost::mpl::at_c<PassedAccessors, Accessor::index_type::value-1>::type(accessor.template get<2>(), accessor.template get<1>(), accessor.template get<0>()));
        }

        template <typename Accessor>
        typename boost::enable_if_c<(Accessor::index_type::value == OutArg), ReturnType>::type&
        operator()(Accessor const&) const {
            std::cout << "Giving the ref (OutArg=" << OutArg << ") " << m_result << std::endl;
            return m_result;
        }

    };

    template <typename Evaluator, typename ResultType, typename Functor, int OutArg>
    struct insert_argument {
        Evaluator const& eval;

        insert_argument(Evaluator const& eval)
            : eval(eval)
        {}

        template <typename ...Args>
        function_aggregator<Evaluator, typename gridtools::variadic_to_vector<Args...>::type, ResultType, OutArg>
        operator()(Args const... args) const &
        {
            return function_aggregator<Evaluator, typename gridtools::variadic_to_vector<Args...>::type, ResultType, OutArg>(eval);
        }
    };

    template <typename Functor>
    struct _get_index {

        template <int I, int L, typename List>
        struct scan_for_index {
            using type = typename boost::mpl::if_<typename std::is_const<typename boost::mpl::at_c<List, I>::type >::type,
                                                  typename scan_for_index<I+1, L, List>::type,
                                                  std::integral_constant<int, I>
                                                  >::type;
        };

        template <int I, typename List>
        struct scan_for_index<I, I, List> {
            using type = std::integral_constant<int,-1>;
        };

        static const int value = scan_for_index<0, boost::mpl::size<typename Functor::arg_list>::value, typename Functor::arg_list>::type::value;
    };

    template <typename Functor>
    constexpr int get_index_of_first_non_const() {
        return _get_index<Functor>::value;
    }

    template <typename Evaluator, typename Functor, typename Region, int OutArg, typename ResultType>
    struct call_the_damn_thing {
        Evaluator const& eval;

        call_the_damn_thing(Evaluator const& eval)
            : eval(eval)
        {}

        template <typename ...Args>
        ResultType then_do_it_damn_it(Args const& ...args) const {
            auto newargs = insert_argument<
                Evaluator,
                ResultType,
                Functor,
                get_index_of_first_non_const<Functor>()>(eval)(args...);
            Functor::Do(newargs, Region());
            return newargs.result();
        }
    };

    template <typename Functor, typename ResultType, typename Region, int Offi=0, int Offj=0, int Offk=0>
    struct call {

        template <typename Evaluator, typename ...Args>
        static ResultType with(Evaluator const& eval, Args const& ...args) {

            // Here several checks can be performed

            return call_the_damn_thing<
                Evaluator,
                Functor,
                Region,
                get_index_of_first_non_const<Functor>(),
                ResultType>(eval)
                .then_do_it_damn_it(args...);
        }
    };

} // namespace gridtools
