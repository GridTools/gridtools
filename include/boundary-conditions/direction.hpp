/*
  GridTools Libraries

  Copyright (c) 2017, GridTools Consortium
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

/**
@file
@brief definition of direction in a 3D cartesian grid
 */
namespace gridtools {
    /**
       @brief Enum defining the directions in a discrete Cartesian grid
     */
    enum sign { any_ = -2, minus_ = -1, zero_, plus_ };
    /**
       @brief Class defining a direction in a cartesian 3D grid.
       The directions correspond to the following:
       - all the three template parameters are either plus or minus: identifies a node on the cell
       \verbatim
       e.g. direction<minus_, plus_, minus_> corresponds to:
         .____.
        /    /|
       o____. |
       |    | .          z
       |    |/       x__/
       .____.           |
                        y
       \endverbatim

       - there is one zero parameter: identifies one edge
       \verbatim
       e.g. direction<zero_, plus_, minus_> corresponds to:
         .____.
        /    /|
       .####. |
       |    | .
       |    |/
       .____.
       \endverbatim

       - there are 2 zero parameters: identifies one face
       \verbatim
       e.g. direction<zero_, zero_, minus_> corresponds to:
         .____.
        /    /|
       .____. |
       |####| .
       |####|/
       .####.
       \endverbatim
       - the case in which all three are zero does not belong to the boundary and is excluded.
     */
    template < sign I_, sign J_, sign K_, class Predicate = boost::enable_if_c< true >::type >
    struct direction {
        static const sign I = I_;
        static const sign J = J_;
        static const sign K = K_;
    };

    template < sign I, sign J, sign K >
    std::ostream &operator<<(std::ostream &s, direction< I, J, K > const &) {
        s << "direction<" << I << ", " << J << ", " << K << ">";
        return s;
    }

} // namespace gridtools
