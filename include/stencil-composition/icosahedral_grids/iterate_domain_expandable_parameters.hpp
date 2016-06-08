#pragma once
#include "../iterate_domain.hpp"

/** @file iterate_domain for expandable parameters*/

namespace gridtools {

    template < typename T >
    struct is_iterate_domain;

    /**
       @brief iterate_domain specific for when expandable parameters are used

       In expandable parameter computations the user function is repeated a specific amount of time in
       each stencil. The parameters are stored in a storage list, and consecutive elements of the list
       are accessed in each user function.
       This struct "decorates" the base iterate_domain instance with a static const integer ID, which
       records the current position in the storage list, and reimplements the operator() in order to
       access the storage list at the correct offset.

       \tparam IterateDomain base iterate_domain class. Might be e.g. iterate_domain_host or iterate_domain_cuda
       \tparam Value the current position in the storage list
     */
    template < typename IterateDomain, ushort_t Value >
    struct iterate_domain_expandable_parameters : public IterateDomain {

#ifdef CXX11_ENABLED
        GRIDTOOLS_STATIC_ASSERT(is_iterate_domain< IterateDomain >::value, "wrong type");
        static const ushort_t ID = Value - 1;
        typedef IterateDomain super;
        typedef IterateDomain iterate_domain_t;

        using super::operator();
        /**
           @brief set the offset in the storage_list and forward to the base class

           when the vector_accessor is passed to the iterate_domain we know we are accessing an
           expandable parameters list. Accepts rvalue arguments (accessors constructed in-place)

           \param arg the vector accessor
         */
        // rvalue
        template < uint_t ACC_ID, enumtype::intend Intent, typename LocationType, typename Extent >
        GT_FUNCTION typename super::iterate_domain_t::template accessor_return_type<
            accessor< ACC_ID, Intent, LocationType, Extent > >::type
        operator()(vector_accessor< ACC_ID, Intent, LocationType, Extent > &&arg) const {
            GRIDTOOLS_STATIC_ASSERT(is_extent< Extent >::value, "wrong type");
            arg.template set< 0 >(ID);

            return super::operator()((accessor< ACC_ID, Intent, LocationType, Extent >)arg);
        }

        /**
           @brief set the offset in the storage_list and forward to the base class

           This version is identical to the previous one, but it accepts an lvalue as argument.

           \param arg the vector accessor
         */
        // lvalue
        template < uint_t ID, enumtype::intend Intent, typename LocationType, typename Extent >
        GT_FUNCTION typename super::iterate_domain_t::template accessor_return_type<
            vector_accessor< ID, Intent, LocationType, Extent > >::type
        operator()(vector_accessor< ID, Intent, LocationType, Extent > &arg) const {
            GRIDTOOLS_STATIC_ASSERT(is_extent< Extent >::value, "wrong type");
            arg.template set< 0 >(ID);
            return super::operator()((accessor< ID, Intent, LocationType, Extent >)arg);
        }

#else // CXX11_ENABLED
        GRIDTOOLS_STATIC_ASSERT(Value,
            "You are using a expandable_parameters and compiling with C++03, switch to C++11 (-DENABLE_CXX11=ON)");
#endif
    };

    template < typename T >
    struct is_iterate_domain_expandable_parameters : boost::mpl::false_ {};

    template < typename T, ushort_t Val >
    struct is_iterate_domain_expandable_parameters< iterate_domain_expandable_parameters< T, Val > >
        : boost::mpl::true_ {};

    template < typename T, ushort_t Val >
    struct is_iterate_domain< iterate_domain_expandable_parameters< T, Val > > : boost::mpl::true_ {};
}
