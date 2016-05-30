#pragma once
namespace gridtools {

    template < ushort_t ID >
    struct gt_get {

        template < typename First, typename... T >
        GT_FUNCTION static constexpr const First apply(First const &first_, T const &... rest_) {
            return (ID == 0) ? first_ : gt_get< ID - 1 >::apply(rest_...);
        }

        template < typename First >
        GT_FUNCTION static constexpr const First apply(First const &first_) {
            return first_;
        }
    };
}//namespace gridtools
