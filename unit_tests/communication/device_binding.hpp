/*
  GridTools Libraries

  Copyright (c) 2016, GridTools Consortium
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  1. Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

  For information: http://eth-cscs.github.io/gridtools/
*/

#pragma once

#ifdef _USE_GPU_
/* device_binding added by Devendar Bureddy, OSU */
void device_binding() {

    int local_rank = 0 /*, num_local_procs*/;
    int dev_count, use_dev_count, my_dev_id;
    char *str;

    printf("HOME %s\n", getenv("HOME"));

    if ((str = getenv("MV2_COMM_WORLD_LOCAL_RANK")) != NULL) {
        local_rank = atoi(str);
        printf("MV2_COMM_WORLD_LOCAL_RANK %s\n", str);
    } else if ((str = getenv("SLURM_LOCALID")) != NULL) {
        local_rank = atoi(str);
        printf("SLURM_LOCALID %s\n", str);
    }

    if ((str = getenv("MPISPAWN_LOCAL_NPROCS")) != NULL) {
        // num_local_procs = atoi (str);
        printf("MPISPAWN_LOCAL_NPROCS %s\n", str);
    }

    cudaGetDeviceCount(&dev_count);
    if ((str = getenv("NUM_GPU_DEVICES")) != NULL) {
        use_dev_count = atoi(str);
        printf("NUM_GPU_DEVICES %s\n", str);
    } else {
        use_dev_count = dev_count;
        printf("NUM_GPU_DEVICES %d\n", use_dev_count);
    }

    my_dev_id = local_rank % use_dev_count;
    printf("local rank = %d dev id = %d\n", local_rank, my_dev_id);
    cudaSetDevice(my_dev_id);
}
#endif
