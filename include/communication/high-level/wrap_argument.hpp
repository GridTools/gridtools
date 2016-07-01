/*
   Copyright 2016 GridTools Consortium

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/
#ifndef _WRAP_ARGUMENT_H_
#define _WRAP_ARGUMENT_H_

struct wrap_argument {
    int data[27];

    __host__ __device__ wrap_argument(int const *ptr) {
        for (int i = 0; i < 27; ++i)
            data[i] = ptr[i];
    }

    __host__ __device__ int &operator[](int i) { return data[i]; }

    __host__ __device__ int const &operator[](int i) const { return data[i]; }
};

#endif
