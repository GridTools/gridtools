/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include <stdbool.h>

enum gt_fortran_array_kind {
    gt_fk_Bool,
    gt_fk_Int,
    gt_fk_Short,
    gt_fk_Long,
    gt_fk_LongLong,
    gt_fk_Float,
    gt_fk_Double,
    gt_fk_LongDouble,
    gt_fk_SignedChar
};
typedef enum gt_fortran_array_kind gt_fortran_array_kind;

struct gt_fortran_array_descriptor {
    gt_fortran_array_kind type;
    int rank;
    int dims[7];
    void *data;
    bool is_acc_present;
    // TODO: add support for strides, bounds end type gt_fortran_array_descriptor
};
typedef struct gt_fortran_array_descriptor gt_fortran_array_descriptor;
