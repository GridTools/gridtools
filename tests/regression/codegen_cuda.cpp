/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <array>

#include <gtest/gtest.h>

#include <gridtools/common/integral_constant.hpp>
#include <gridtools/common/tuple_util.hpp>
#include <gridtools/sid/allocator.hpp>
#include <gridtools/sid/as_const.hpp>
#include <gridtools/sid/composite.hpp>
#include <gridtools/sid/concept.hpp>
#include <gridtools/sid/contiguous.hpp>
#include <gridtools/sid/loop.hpp>
#include <gridtools/sid/sid_shift_origin.hpp>
#include <gridtools/stencil/common/dim.hpp>
#include <gridtools/storage/builder.hpp>
#include <gridtools/storage/>
#include <gridtools/storage/sid.hpp>

#include <verifier.hpp>

#include "horizontal_diffusion_repository.hpp"

using namespace gridtools;
using namespace literals;
namespace dim = stencil::dim;

struct domain {
    std::array<int, 3> size;
    std::array<int, 3> origin;
};

template <class Key, class Dim, class Ptr, class Strides, class Offset>
auto shifted(Ptr const &ptr, Strides const &strides, Offset offset) {
    auto res = at_key<Key>(ptr);
    sid::shift(res, sid::get_stride_element<Key, Dim>(strides), offset);
    return res;
}

struct coeff;
struct in;
struct out;
struct lap;
struct flx;
struct fly;

auto const lap_stencil = [](auto const &ptr, auto const &strides) {
  *at_key<lap>(ptr) = 4. * *at_key<in>(ptr) -
                      (*shifted<in, dim::i>(ptr, strides, 1_c) + *shifted<in, dim::j>(ptr, strides, 1_c) +
                       *shifted<in, dim::i>(ptr, strides, -1_c) + *shifted<in, dim::j>(ptr, strides, -1_c));
};

auto const flx_stencil = [](auto const &ptr, auto const &strides) {
  auto res = *shifted<lap, dim::i>(ptr, strides, 1_c) - *at_key<lap>(ptr);
  *at_key<flx>(ptr) = res * (*shifted<in, dim::i>(ptr, strides, 1_c) - *at_key<in>(ptr)) > 0 ? 0 : res;
};

auto const fly_stencil = [](auto const &ptr, auto const &strides) {
  auto res = *shifted<lap, dim::j>(ptr, strides, 1_c) - *at_key<lap>(ptr);
  *at_key<fly>(ptr) = res * (*shifted<in, dim::j>(ptr, strides, 1_c) - *at_key<in>(ptr)) > 0 ? 0 : res;
};

auto const out_stencil = [](auto &ptr, auto const &strides) {
  *at_key<out>(ptr) =
      *at_key<in>(ptr) - *at_key<coeff>(ptr) * (*at_key<flx>(ptr) - *shifted<flx, dim::i>(ptr, strides, -1_c) +
                                                *at_key<fly>(ptr) - *shifted<fly, dim::j>(ptr, strides, -1_c));
};

auto hori_diff(domain const &dom) {
    struct tmp_stride_kind;

    return [=](auto &&raw_coeff, auto &&raw_in, auto &&raw_out) {
      static_assert(is_sid<decltype(raw_coeff)>(), "");
      static_assert(is_sid<decltype(raw_in)>(), "");
      static_assert(is_sid<decltype(raw_out)>(), "");

      auto alloc = sid::make_cached_allocator(&std::make_unique<char[]>);

      auto make_tmp = [&] {
        auto size = tuple_util::make<std::array>(dom.size[0] + 2, dom.size[1] + 2, dom.size[2] + 2);
        auto offset = tuple_util::make<std::array>(1, 1, 0);
        return sid::shift_sid_origin(sid::make_contiguous<double, ptrdiff_t, tmp_stride_kind>(alloc, size), offset);
      };

      auto shift = [&](auto &&sid) { return sid::shift_sid_origin(std::forward<decltype(sid)>(sid), dom.origin); };

      auto data = tuple_util::make<sid::composite::keys<coeff, in, out, lap, flx, fly>::values>(
          sid::as_const(shift(std::forward<decltype(raw_coeff)>(raw_coeff))),
          sid::as_const(shift(std::forward<decltype(raw_in)>(raw_in))),
          shift(std::forward<decltype(raw_out)>(raw_out)),
          make_tmp(),
          make_tmp(),
          make_tmp());

      auto origin = sid::get_origin(data);
      auto strides = sid::get_strides(data);

      auto run = [&](auto const &stencil, auto iminus, auto iplus, auto jminus, auto jplus) {
        auto ptr = origin();
        sid::shift(ptr, sid::get_stride<dim::i>(strides), iminus);
        sid::shift(ptr, sid::get_stride<dim::j>(strides), jminus);
        auto i_loop = sid::make_loop<dim::i>(dom.size[0] + iplus - iminus);
        auto j_loop = sid::make_loop<dim::j>(dom.size[1] + jplus - jminus);
        auto k_loop = sid::make_loop<dim::k>(dom.size[2]);
        i_loop(j_loop(k_loop(stencil)))(ptr, strides);
      };

      run(lap_stencil, -1_c, 1_c, -1_c, 1_c);
      run(flx_stencil, -1_c, 0_c, 0_c, 0_c);
      run(fly_stencil, 0_c, 0_c, -1_c, 0_c);
      run(out_stencil, 0_c, 0_c, 0_c, 0_c);
    };
}

TEST(codegen_naive, hori_diff) {
constexpr int I = 9, J = 9, K = 5, Halo = 2;
horizontal_diffusion_repository repo(I, J, K);
auto builder = storage::builder<storage::cpu_ifirst>.type<double>().dimensions(I, J, K);
auto out = builder.build();
hori_diff({{I - 4, J - 4, K}, {Halo, Halo}})(
builder.initializer(repo.coeff).build(), builder.initializer(repo.in).build(), out);
size_t halos[3][2] = {{Halo, Halo}, {Halo, Halo}};
EXPECT_TRUE(verify_data_store(repo.out, out, halos));
}