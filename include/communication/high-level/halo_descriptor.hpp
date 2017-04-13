/*
  GridTools Libraries

  Copyright (c) 2017, ETH Zurich and MeteoSwiss
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
#ifndef _HALO_DESCRIPTOR_H_
#define _HALO_DESCRIPTOR_H_

#include <common/gt_assert.hpp>
#include "../../common/array.hpp"

namespace gridtools {

    /** \class halo_descriptor
        Given a dimension it is described like this:
        \code
        |-----|------|---------------|---------|----|
        | pad0|minus |    length     | plus    |pad1|
                      ^begin        ^end
        |               total_length                |
        \endcode

        In this way five numbers are needed to describe the whole
        structure, assuming the first index of pad0 is fixed to 0, and
        they are minus, begin, end, plus, and total_length.
    */
    struct halo_descriptor {
      private:
        int m_minus;        // halo on the minus direction
        int m_plus;         // halo on the plus direction
        int m_begin;        // index of the fisrt element of the active region
        int m_end;          // index of the last element of the active region
        int m_total_length; // minus+plus+(end-begin+1)+pads

      public:
        /** Default constructors: all parameters to zero. Used to be able
            to have vectors of halo descriptors with initialized values
            that are subsequently copy constructed (since this is a POD
            structure it can be copied without an explicit copy
            constructor.
         */
        __host__ __device__ halo_descriptor() : m_minus(0), m_plus(0), m_begin(0), m_end(0), m_total_length(0) {}

        /** Main constructors where the five integers are passed.
          \code
          |-----|------|---------------|---------|----|
          | pad0|minus |    length     | plus    |pad1|
                        ^begin        ^end
          |               total_length                |
          \endcode

          \param[in] m The minus parameter
          \param[in] p The plus parameter
          \param[in] b The begin parameter
          \param[in] e The end parameter (inclusive)
          \param[in] l The total_length parameter
         */
        __host__ __device__ halo_descriptor(int m, int p, int b, int e, int l)
            : m_minus(m), m_plus(p), m_begin(b), m_end(e), m_total_length(l) {}

        __host__ __device__ halo_descriptor &operator=(halo_descriptor const &hh) {
            m_minus = hh.minus();
            m_plus = hh.plus();
            m_begin = hh.begin();
            m_end = hh.end();
            m_total_length = hh.total_length();
            return *this;
        }

        __host__ __device__ halo_descriptor(halo_descriptor const &hh) {
            m_minus = hh.minus();
            m_plus = hh.plus();
            m_begin = hh.begin();
            m_end = hh.end();
            m_total_length = hh.total_length();
        }

        /**
           Begin index for the loop on the outside region.
           \param[in] I relative coordinate of the neighbor (the \f$eta\f$ parameter in \link MULTI_DIM_ACCESS \endlink)
         */
        __host__ __device__ int loop_low_bound_outside(
            int I) const { // inside is the fact that the halos are ones outside the begin-end region
            if (I == 0)
                return m_begin;
            if (I == 1)
                return m_end + 1;
            if (I == -1)
                return m_begin - m_minus;

            assert(false);
            return 0;
        }

        /**
           End index for the loop on the outside region.
           \param[in] I relative coordinate of the neighbor (the \f$eta\f$ parameter in \link MULTI_DIM_ACCESS \endlink)
         */
        __host__ __device__ int loop_high_bound_outside(
            int I) const { // inside is the fact that the halos are ones outside the begin-end region
            if (I == 0)
                return m_end;
            if (I == 1)
                return m_end + m_plus;
            if (I == -1)
                return m_begin - 1;

            assert(false);
            return 0;
        }

        /**
           Begin index for the loop on the inside region.
           \param[in] I relative coordinate of the neighbor (the \f$eta\f$ parameter in \link MULTI_DIM_ACCESS \endlink)
         */
        __host__ __device__ int loop_low_bound_inside(
            int I) const { // inside is the fact that the halos are ones outside the begin-end region
            if (I == 0)
                return m_begin;
            if (I == 1)
                return m_end - m_minus + 1;
            if (I == -1)
                return m_begin;

            assert(false);
            return 0;
        }

        /**
           End index for the loop on the inside region.
           \param[in] I relative coordinate of the neighbor (the \f$eta\f$ parameter in \link MULTI_DIM_ACCESS \endlink)
         */
        __host__ __device__ int loop_high_bound_inside(
            int I) const { // inside is the fact that the halos are ones outside the begin-end region
            if (I == 0)
                return m_end;
            if (I == 1)
                return m_end;
            if (I == -1)
                return m_begin + m_plus - 1;

            assert(false);
            return 0;
        }

        /**
           receive_length. Return the parameter \f$S_i\f$ as described in \link MULTI_DIM_ACCESS \endlink.

           \param[in] I relative coordinate of the neighbor (the \f$eta\f$ parameter in \link MULTI_DIM_ACCESS \endlink)
        */
        __host__ __device__ int r_length(int I) const {
            switch (I) {
            case 0:
                return (m_end - m_begin + 1);
            case 1:
                return m_plus;
            case -1:
                return m_minus;
            default:
                assert(false);
                return -1;
            }
        }

        /**
           send_length. Return the parameter \f$S_i\f$ as described in \link MULTI_DIM_ACCESS \endlink.

           \param[in] I relative coordinate of the neighbor (the \f$eta\f$ parameter in \link MULTI_DIM_ACCESS \endlink)
        */
        __host__ __device__ int s_length(int I) const {
            switch (I) {
            case 0:
                return (m_end - m_begin + 1);
            case -1:
                return m_plus;
            case 1:
                return m_minus;
            default:
                assert(false);
                return -1;
            }
        }

        __host__ __device__ int minus() const { return m_minus; }
        __host__ __device__ int plus() const { return m_plus; }
        __host__ __device__ int begin() const { return m_begin; }
        __host__ __device__ int end() const { return m_end; }
        __host__ __device__ int total_length() const { return m_total_length; }

        __host__ __device__ void set_minus(int value) { m_minus = value; }
        __host__ __device__ void set_plus(int value) { m_plus = value; }
        __host__ __device__ void set_begin(int value) { m_begin = value; }
        __host__ __device__ void set_end(int value) { m_end = value; }
        __host__ __device__ void set_total_length(int value) { m_total_length = value; }
    };

    inline std::ostream &operator<<(std::ostream &s, halo_descriptor const &hd) {
        return s << "hd(" << hd.minus() << ", " << hd.plus() << ", " << hd.begin() << ", " << hd.end() << ", "
                 << hd.total_length() << ")";
    }
} // namespace gridtools

#endif
