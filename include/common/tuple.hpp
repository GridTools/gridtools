#pragma once
#include "host_device.hpp"
#include "generic_metafunctions/variadic_typedef.hpp"

namespace gridtools {

    namespace _impl {
        template < ushort_t Idx, typename Args, typename First, typename Super >
        struct return_helper {
            GT_FUNCTION constexpr typename Args::template get_elem< Idx >::type operator()(
                const First f, const Super x) {
                return x.template get< Idx - 1 >();
            }
        };

        template < typename Args, typename First, typename Super >
        struct return_helper< 0, Args, First, Super > {
            GT_FUNCTION constexpr First operator()(const First f, const Super x) { return f; }
        };
    }

    template < typename... Args >
    struct tuple;

    template < typename First, typename... Args >
    struct tuple< First, Args... > : public tuple< Args... > {

        typedef tuple< First, Args... > type;
        typedef variadic_typedef< First, Args... > args_t;
        typedef tuple< Args... > super;

        static const size_t n_dimensions = sizeof...(Args) + 1;

        /**@brief constructor taking an integer as the first argument, and then other optional arguments.
           The integer gets assigned to the current extra dimension and the other arguments are passed to the base class
           (in order to get assigned to the other dimensions).
           When this constructor is used all the arguments have to be specified and passed to the function call in
           order. No check is done on the order*/
        GT_FUNCTION constexpr tuple(const First t, Args const... x) : super(x...), m_offset(t) {}

        GT_FUNCTION constexpr First operator()() { return m_offset; }

        /**@brief returns the offset at a specific index Idx*/
        template < ushort_t Idx >
        /**@brief returns the offset array*/
        GT_FUNCTION constexpr typename args_t::template get_elem< Idx >::type get() const {

            typedef _impl::return_helper< Idx, args_t, First, super > helper;
            return helper()(m_offset, *this);
        }

      protected:
        First m_offset;
    };

    template < typename First >
    struct tuple< First > {

        static const size_t n_dimensions = 1;

        typedef tuple< First > type;
        typedef variadic_typedef< First > args_t;

        template < ushort_t Idx >
        struct get_elem {
            typedef typename args_t::template get_elem< Idx >::type type;
        };

        /**@brief constructor taking an integer as the first argument, and then other optional arguments.
           The integer gets assigned to the current extra dimension and the other arguments are passed to the base class
           (in order to get assigned to the other dimensions).
           When this constructor is used all the arguments have to be specified and passed to the function call in
           order. No check is done on the order*/
        GT_FUNCTION constexpr tuple(const First t) : m_offset(t) {}

        GT_FUNCTION constexpr First operator()() { return m_offset; }

        /**@brief returns the offset at a specific index Idx*/
        template < ushort_t Idx >
        GT_FUNCTION constexpr typename get_elem< Idx >::type get() const {
            GRIDTOOLS_STATIC_ASSERT((Idx == 0), "Error: out of bound tuple access");
            return m_offset;
        }

      protected:
        First m_offset;
    };

    template < typename T >
    struct is_tuple : boost::mpl::false_ {};

    template < typename... Args >
    struct is_tuple< tuple< Args... > > : boost::mpl::true_ {};

    template < typename... Args >
    tuple< Args... > make_tuple(Args... args) {
        return tuple< Args... >(args...);
    }

} // namespace gridtools
