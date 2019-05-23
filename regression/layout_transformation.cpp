/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gridtools/interface/layout_transformation/layout_transformation.hpp>
#include <gridtools/tools/regression_fixture.hpp>

#include <gtest/gtest.h>

using namespace gridtools;

namespace {
    class generic_benchmark {
      public:
        template <typename F>
        generic_benchmark(const std::string &name, F &&f) : m_f(f), m_meter(name) {}

        void run() {
            m_meter.start();
            m_f();
            m_meter.pause();
        }
        void reset_meter() { m_meter.reset(); }
        std::string print_meter() { return m_meter.to_string(); }

      private:
        std::function<void()> m_f;

        using performance_meter_t = typename timer_traits<backend_t>::timer_type;
        performance_meter_t m_meter;
    };

    template <typename Src, typename Dst>
    void transform(Src &src, Dst &dst) {
        auto src_v = gridtools::make_target_view(src);
        auto dst_v = gridtools::make_target_view(dst);

        storage_info_rt si_src = make_storage_info_rt(*src.get_storage_info_ptr());
        storage_info_rt si_dst = make_storage_info_rt(*dst.get_storage_info_ptr());

        gridtools::interface::transform(
            dst_v.data(), src_v.data(), si_src.total_lengths(), si_dst.strides(), si_src.strides());
    }
    template <typename Src, typename Dst>
    void verify_result(Src &src, Dst &dst) {
        src.sync();
        dst.sync();

        auto src_v = gridtools::make_host_view<access_mode::read_only>(src);
        auto dst_v = gridtools::make_host_view<access_mode::read_only>(dst);

        for (int i = 0; i < src.template total_length<0>(); ++i)
            for (int j = 0; j < src.template total_length<1>(); ++j)
                for (int k = 0; k < src.template total_length<2>(); ++k)
                    EXPECT_EQ(src_v(i, j, k), dst_v(i, j, k));

        src.sync();
        dst.sync();
    }
} // namespace

template <typename SrcLayout, typename SrcAlignment, typename DstLayout, typename DstAlignment>
struct layout_transformation : regression_fixture<0> {
  private:
    template <int_t Id, typename Layout, typename Alignment>
    using storage_info_t =
        typename storage_tr::select_custom_layout_storage_info_align<Id, Layout, halo<0, 0, 0>, Alignment>::type;
    template <int_t Id, typename Layout, typename Alignment>
    using storage_t = storage_tr::data_store_t<float_type, storage_info_t<Id, Layout, Alignment>>;

    using src_storage_t = storage_t<0, SrcLayout, SrcAlignment>;
    using dst_storage_t = storage_t<1, DstLayout, DstAlignment>;

    src_storage_t src = make_storage<src_storage_t>([](int i, int j, int k) { return i + j + k; });
    dst_storage_t dst = make_storage<dst_storage_t>(-1.);

  public:
    void run_test() {

        transform(src, dst);
        verify_result(src, dst);

        benchmark(generic_benchmark{"same_layout", [&]() { transform(src, dst); }});
    }
};

using IJK_to_IJK = layout_transformation<layout_map<0, 1, 2>, alignment<1>, layout_map<0, 1, 2>, alignment<32>>;
TEST_F(IJK_to_IJK, perf) { run_test(); }

using IJK_to_KJI = layout_transformation<layout_map<0, 1, 2>, alignment<1>, layout_map<2, 1, 0>, alignment<1>>;
TEST_F(IJK_to_KJI, perf) { run_test(); }
