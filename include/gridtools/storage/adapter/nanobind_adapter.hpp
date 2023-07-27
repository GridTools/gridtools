#pragma once

#include "../../common/array.hpp"
#include "../../common/integral_constant.hpp"
#include "../../sid/simple_ptr_holder.hpp"
#include "../../sid/synthetic.hpp"
#include "../../sid/unknown_kind.hpp"
#include <algorithm>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace gridtools {
    namespace nanobind_sid_adapter_impl_ {
        template <size_t, class>
        struct kind {};

        template <std::size_t UnitStrideDim = std::size_t(-1), class T, std::size_t... Sizes, class... Args>
        auto as_sid(nanobind::ndarray<T, nanobind::shape<Sizes...>, Args...> ndarray) {
            using sid::property;
            const auto ptr = ndarray.data();
            constexpr auto ndim = sizeof...(Sizes);
            assert(ndim == ndarray.ndim());
            gridtools::array<size_t, ndim> shape;
            gridtools::array<size_t, ndim> strides;
            std::copy_n(ndarray.shape_ptr(), ndim, shape.begin());
            std::copy_n(ndarray.stride_ptr(), ndim, strides.begin());

            return sid::synthetic()
                .template set<property::origin>(sid::host_device::simple_ptr_holder<T *>{ptr})
                .template set<property::strides>(strides)
                .template set<property::strides_kind, sid::unknown_kind>()
                .template set<property::lower_bounds>(gridtools::array<integral_constant<size_t, 0>, ndim>())
                .template set<property::upper_bounds>(shape);
        }
    } // namespace nanobind_sid_adapter_impl_

    using nanobind_sid_adapter_impl_::as_sid;
} // namespace gridtools