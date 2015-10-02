#pragma once
namespace gridtools{

    struct gt_get{
        template<ushort_t ID, typename First, typename ... T>
        GT_FUNCTION
        static constexpr  const First apply(First const& first_, T const& ... rest_) {
            return (ID==0) ? first_ : gt_get::template apply<ID-1>(rest_ ...);
        }

        template<ushort_t ID, typename First>
        GT_FUNCTION
        static constexpr const First apply(First const& first_) {
            // GRIDTOOLS_STATIC_ASSERT(ID==0, "access out of bound in a call to gt_get");
            return first_;
        }
    };
}
