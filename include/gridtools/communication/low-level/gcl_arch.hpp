/*
 * GridTools
 *
 * Copyright (c) 2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

/** \file
    In this file the different types of architectures are
    defined. The Architectures specifies where the data to be
    exchanged by communication patterns are residing. The possible
    choices are: \link gridtools::gcl_cpu \endlink , \link gridtools::gcl_gpu
    \endlink , end, for illustration purpose only, \link gridtools::gcl_mc
    \endlink , which is not supported at the moment.

    The assumption is that data to be exchanged is in the same place
    for all the processes involved in a pattern. That is, it is not
    possible to send data from a cpu main memory to a remote GPU
    memory.
*/

namespace gridtools {
    /** Indicate that the data is on the main memory of the process
     */
    struct gcl_cpu {};

    /** Indicates that the data is on the memory of a GPU
     */
    struct gcl_gpu {};

    /** Indicates that the data is on the memory of a MC card.

        Note: Not supported, placed here only for illustration.
     */
    struct gcl_mc {}; // Not supported, placed here only for illustration.
} // namespace gridtools
