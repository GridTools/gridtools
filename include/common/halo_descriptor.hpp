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
#ifndef _HALO_DESCRIPTOR_H_
#define _HALO_DESCRIPTOR_H_

#include "array.hpp"

/**
@file
@brief defines the halo region
*/

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
        uint_t m_minus;        // halo on the minus direction
        uint_t m_plus;         // halo on the plus direction
        uint_t m_begin;        // index of the fisrt element of the active region
        uint_t m_end;          // index of the last element of the active region
        uint_t m_total_length; // minus+plus+(end-begin+1)+pads

      public:
        /** Default constructors: all parameters to zero. Used to be able
            to have vectors of halo descriptors with initialized values
            that are subsequently copy constructed (since this is a POD
            structure it can be copied without an explicit copy
            constructor.
         */
        GT_FUNCTION halo_descriptor() : m_minus(0), m_plus(0), m_begin(0), m_end(0), m_total_length(0) {}

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
        GT_FUNCTION halo_descriptor(uint_t m, uint_t p, uint_t b, uint_t e, uint_t l)
            : m_minus(m), m_plus(p), m_begin(b), m_end(e), m_total_length(l) {}

        GT_FUNCTION halo_descriptor &operator=(halo_descriptor const &hh) {
            m_minus = hh.minus();
            m_plus = hh.plus();
            m_begin = hh.begin();
            m_end = hh.end();
            m_total_length = hh.total_length();
            return *this;
        }

        GT_FUNCTION halo_descriptor(halo_descriptor const &hh) {
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
        GT_FUNCTION int_t loop_low_bound_outside(
            short_t I) const { // inside is the fact that the halos are ones outside the begin-end region
            if (I == 0)
                return m_begin;
            if (I == 1)
                return m_end + 1;
            if (I == -1) {
                assert((int_t)(m_begin - m_minus) >= 0);
                return (m_begin - m_minus);
            }

            assert(false);
            return 1;
        }

        /**
           End index for the loop on the outside region.
           \param[in] I relative coordinate of the neighbor (the \f$eta\f$ parameter in \link MULTI_DIM_ACCESS \endlink)
         */
        GT_FUNCTION int_t loop_high_bound_outside(
            short_t I) const { // inside is the fact that the halos are ones outside the begin-end region
            if (I == 0)
                return m_end;
            if (I == 1)
                return m_end + m_plus;
            if (I == -1) {
                //  assert(m_begin-1>=0);
                return m_begin - 1;
            }

            assert(false);
            return 1;
        }

        /**
           Begin index for the loop on the inside region.
           \param[in] I relative coordinate of the neighbor (the \f$eta\f$ parameter in \link MULTI_DIM_ACCESS \endlink)
         */
        GT_FUNCTION int_t loop_low_bound_inside(
            short_t I) const { // inside is the fact that the halos are ones outside the begin-end region
            if (I == 0)
                return m_begin;
            if (I == 1)
                return m_end - m_minus + 1;
            if (I == -1)
                return m_begin;

            assert(false);
            return 1;
        }

        /**
           End index for the loop on the inside region.
           \param[in] I relative coordinate of the neighbor (the \f$eta\f$ parameter in \link MULTI_DIM_ACCESS \endlink)
         */
        GT_FUNCTION int_t loop_high_bound_inside(
            short_t I) const { // inside is the fact that the halos are ones outside the begin-end region
            if (I == 0)
                return m_end;
            if (I == 1)
                return m_end;
            if (I == -1) {
                //  assert(m_begin+m_plus-1>=0);
                return m_begin + m_plus - 1;
            }

            assert(false);
            return 1;
        }

        /**
           receive_length. Return the parameter \f$S_i\f$ as described in \link MULTI_DIM_ACCESS \endlink.

           \param[in] I relative coordinate of the neighbor (the \f$eta\f$ parameter in \link MULTI_DIM_ACCESS \endlink)
        */
        GT_FUNCTION uint_t r_length(short_t I) const {
            switch (I) {
            case 0:
                return (m_end - m_begin + 1);
            case 1:
                return m_plus;
            case -1:
                return m_minus;
            default:
                assert(false);
                return 1;
            }
        }

        /**
           send_length. Return the parameter \f$S_i\f$ as described in \link MULTI_DIM_ACCESS \endlink.

           \param[in] I relative coordinate of the neighbor (the \f$eta\f$ parameter in \link MULTI_DIM_ACCESS \endlink)
        */
        GT_FUNCTION uint_t s_length(short_t I) const {
            switch (I) {
            case 0:
                return (m_end - m_begin + 1);
            case -1:
                return m_plus;
            case 1:
                return m_minus;
            default:
                assert(false);
                return 1;
            }
        }

        GT_FUNCTION uint_t minus() const { return m_minus; }
        GT_FUNCTION uint_t plus() const { return m_plus; }
        GT_FUNCTION uint_t begin() const { return m_begin; }
        GT_FUNCTION uint_t end() const { return m_end; }
        GT_FUNCTION uint_t total_length() const { return m_total_length; }

        GT_FUNCTION void set_minus(uint_t value) { m_minus = value; }
        GT_FUNCTION void set_plus(uint_t value) { m_plus = value; }
        GT_FUNCTION void set_begin(uint_t value) { m_begin = value; }
        GT_FUNCTION void set_end(uint_t value) { m_end = value; }
        GT_FUNCTION void set_total_length(uint_t value) { m_total_length = value; }
    };

    inline std::ostream &operator<<(std::ostream &s, halo_descriptor const &hd) {
        return s << "hd(" << hd.minus() << ", " << hd.plus() << ", " << hd.begin() << ", " << hd.end() << ", "
                 << hd.total_length() << ")";
    }
} // namespace GCL

#endif
