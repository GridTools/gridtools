/*
 * GridTools Libraries
 * Copyright (c) 2019, ETH Zurich
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

namespace gridtools {
    template <typename DataType, int DIMS, typename>
    class hndlr_descriptor_ut;

    template <typename Datatype, typename GridType, typename, typename, typename, int>
    class hndlr_dynamic_ut;

    template <int DIMS,
        typename Haloexch,
        typename proc_layout_abs = typename default_layout_map<DIMS>::type,
        typename Gcl_Arch = gcl_cpu,
        int = version_mpi_pack>
    class hndlr_generic;

    template <typename DataType, typename layoutmap, template <typename> class traits>
    struct field_on_the_fly;

    template <int DIMS, typename Haloexch, typename proc_layout, typename Gcl_Arch, int versiono>
    class hndlr_generic;
} // namespace gridtools
