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

struct gt_handle;

#ifdef __cplusplus

extern "C" void gt_release(gt_handle const *);

#else

typedef struct gt_handle gt_handle;
void gt_release(gt_handle *);

#endif
