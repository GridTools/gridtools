#pragma once
#include "../iterate_domain.hpp"
#include "../../common/generic_metafunctions/static_decorator.hpp"

namespace gridtools{
    template <typename IterateDomain, ushort_t Value>
    struct iterate_domain_expandable_parameters : public IterateDomain {

        static const ushort_t ID=Value-1;
        typedef IterateDomain super;

        //rvalue
        template < uint_t ID, enumtype::intend Intent, typename Extent, uint_t Size >
        GT_FUNCTION typename super::template accessor_return_type< vector_accessor<ID, Intent, Extent, Size> >::type operator()(vector_accessor<ID, Intent, Extent, Size> && arg) const
        {
            typedef vector_accessor<ID, Intent, Extent, Size> vec_t;
            GRIDTOOLS_STATIC_ASSERT((is_vector_accessor< vec_t >::value), "invalid expression");
            arg.template set<vec_t::n_dim-1>(ID);
            return super::operator()((accessor<ID, Intent, Extent, Size>) arg);
        }


        //lvalue
        template < uint_t ID, enumtype::intend Intent, typename Extent, uint_t Size >
        GT_FUNCTION typename super::template accessor_return_type< vector_accessor<ID, Intent, Extent, Size> >::type operator()(vector_accessor<ID, Intent, Extent, Size> & arg) const
        {
            typedef vector_accessor<ID, Intent, Extent, Size> vec_t;
            GRIDTOOLS_STATIC_ASSERT((is_vector_accessor< vec_t >::value), "invalid expression");
            arg.template set<vec_t::n_dim-1>(ID);
            return super::operator()((accessor<ID, Intent, Extent, Size>) arg);
        }


    };
}
