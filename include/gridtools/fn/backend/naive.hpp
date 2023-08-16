/*
 * GridTools
 *
 * Copyright (c) 2014-2023, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include "../../common/functional.hpp"
#include "../../common/hymap.hpp"
#include "../../common/tuple_util.hpp"
#include "../../sid/allocator.hpp"
#include "../../sid/concept.hpp"
#include "../../sid/contiguous.hpp"
#include "../../sid/multi_shift.hpp"
#include "../../sid/unknown_kind.hpp"
#include "../../thread_pool/concept.hpp"
#include "../../thread_pool/dummy.hpp"
#include "../../thread_pool/omp.hpp"
#include "./common.hpp"

namespace gridtools::fn::backend {
    namespace naive_impl_ {
        template <class ThreadPool>
        struct naive_with_threadpool {};

        using naive = naive_with_threadpool<
#if defined(_OPENMP) || defined(GT_HIP_OPENMP_WORKAROUND)
            thread_pool::omp
#else
            thread_pool::dummy
#endif
            >;

        template <class ThreadPool, class Sizes, class Dims = meta::rename<hymap::keys, get_keys<Sizes>>>
        auto make_parallel_loops(ThreadPool, Sizes const &sizes) {
            return [=](auto f) {
                return [=](auto ptr, auto const &strides) {
                    auto loop_f = [&](auto... indices) {
                        auto local_ptr = ptr;
                        sid::multi_shift(local_ptr, strides, Dims::make_values(indices...));
                        f(local_ptr, strides);
                    };

                    tuple_util::apply(
                        [&](auto... sizes) { thread_pool::parallel_for_loop(ThreadPool(), loop_f, int(sizes)...); },
                        sizes);
                };
            };
        }

        template <class ThreadPool, class Sizes, class StencilStage, class MakeIterator, class Composite>
        void apply_stencil_stage(naive_with_threadpool<ThreadPool>,
            Sizes const &sizes,
            StencilStage,
            MakeIterator &&make_iterator,
            Composite &&composite) {
            auto ptr = sid::get_origin(std::forward<Composite>(composite))();
            auto strides = sid::get_strides(std::forward<Composite>(composite));
            make_parallel_loops(ThreadPool(), sizes)([make_iterator = make_iterator()](auto ptr, auto const &strides) {
                StencilStage()(make_iterator, ptr, strides);
            })(ptr, strides);
        }

        template <class ThreadPool,
            class Sizes,
            class ColumnStage,
            class MakeIterator,
            class Composite,
            class Vertical,
            class Seed>
        void apply_column_stage(naive_with_threadpool<ThreadPool>,
            Sizes const &sizes,
            ColumnStage,
            MakeIterator &&make_iterator,
            Composite &&composite,
            Vertical,
            Seed seed) {
            auto ptr = sid::get_origin(std::forward<Composite>(composite))();
            auto strides = sid::get_strides(std::forward<Composite>(composite));
            auto v_size = at_key<Vertical>(sizes);
            make_parallel_loops(ThreadPool(), hymap::canonicalize_and_remove_key<Vertical>(sizes))(
                [v_size = std::move(v_size), make_iterator = make_iterator(), seed = std::move(seed)](auto ptr,
                    auto const &strides) { ColumnStage()(seed, v_size, make_iterator, std::move(ptr), strides); })(
                ptr, strides);
        }

        template <class ThreadPool>
        inline auto tmp_allocator(naive_with_threadpool<ThreadPool> be) {
            return std::make_tuple(be, sid::allocator(&std::make_unique<char[]>));
        }

        template <class ThreadPool, class Allocator, class Sizes, class T>
        auto allocate_global_tmp(
            std::tuple<naive_with_threadpool<ThreadPool>, Allocator> &alloc, Sizes const &sizes, data_type<T>) {
            return sid::make_contiguous<T, int_t, sid::unknown_kind>(std::get<1>(alloc), sizes);
        }
    } // namespace naive_impl_

    using naive_impl_::naive;
    using naive_impl_::naive_with_threadpool;

    using naive_impl_::apply_column_stage;
    using naive_impl_::apply_stencil_stage;

    using naive_impl_::allocate_global_tmp;
    using naive_impl_::tmp_allocator;
} // namespace gridtools::fn::backend
