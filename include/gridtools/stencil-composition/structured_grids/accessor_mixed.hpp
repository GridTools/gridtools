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

#pragma once
#include "../../common/defs.hpp"
#include "../../common/dimension.hpp"
#include "../../common/host_device.hpp"
#include "../../common/generic_metafunctions/meta.hpp"
#include "../accessor_fwd.hpp"
#include "../accessor_metafunctions.hpp"

namespace gridtools {

    template < ushort_t, int_t >
    struct pair_;

    /**@brief same as accessor but mixing run-time offsets with compile-time ones

       When we know beforehand that the dimension which we are querying is
       a compile-time one, we can use the static method get_constexpr() to get the offset.
       Otherwise the method get() checks before among the static dimensions, and if the
       queried dimension is not found it looks up in the dynamic dimensions. Note that this
       lookup is anyway done at compile time, i.e. the get() method returns in constant time.
     */
    template < class Base, class... Pairs >
    struct accessor_mixed;

    template < class Base, ushort_t... Inxs, int_t... Vals >
    struct accessor_mixed< Base, pair_< Inxs, Vals >... > : Base {
        template < class... Ts >
        GT_FUNCTION explicit constexpr accessor_mixed(Ts... args)
            : Base(dimension< Inxs >(Vals)..., args...) {}
        template < class OtherBase >
        GT_FUNCTION constexpr accessor_mixed(accessor_mixed< OtherBase, pair_< Inxs, Vals >... > const &src)
            : Base(src) {}

      private:
        template < ushort_t I >
        using key_t = std::integral_constant< ushort_t, I >;

        template < int_t Val >
        using val_t = std::integral_constant< int_t, Val >;

        using offset_map_t = meta::list< meta::list< key_t< Base::n_dimensions - Inxs >, val_t< Vals > >... >;

        template < int_t I >
        using find_t = GT_META_CALL(meta::mp_find, (offset_map_t, key_t< I >));

      public:
        template < ushort_t I, class Found = find_t< I > >
        GT_FUNCTION constexpr typename std::enable_if< !std::is_void< Found >::value, int_t >::type get() const {
            return meta::lazy::second< Found >::type::value;
        }
        template < ushort_t I, class Found = find_t< I > >
        GT_FUNCTION constexpr typename std::enable_if< std::is_void< Found >::value, int_t >::type get() const {
            return Base::template get< I >();
        }
    };

    /**
       @brief this struct allows the specification of SOME of the arguments before instantiating the offset_tuple.
       It is a language keyword. Usage examples can be found in the unit test \ref accessor_tests.hpp.
       Possible interfaces:
       - runtime alias
\verbatim
alias<arg_t, dimension<3> > field1(-3); //records the offset -3 as dynamic value
\endverbatim
       field1(args...) is then equivalent to arg_t(dimension<3>(-3), args...)
       - compiletime alias
\verbatim
        using field1 = alias<arg_t, dimension<7> >::set<-3>;
\endverbatim
       field1(args...) is then equivalent to arg_t(dimension<7>(-3), args...)

       NOTE: noone checks that you did not specify the same dimension twice. If that happens, the first occurrence of
the dimension is chosen
    */
    template < typename AccessorType, typename... Known >
    struct alias;

    template < typename AccessorType, ushort_t... Inxs >
    struct alias< AccessorType, dimension< Inxs >... > {
        GRIDTOOLS_STATIC_ASSERT(is_accessor< AccessorType >::value,
            "wrong type. If you want to generalize the alias "
            "to something more generic than an offset_tuple "
            "remove this assert.");

        /**
           @brief compile-time aliases, the offsets specified in this way are assured to be compile-time

           This type alias allows to embed some of the offsets directly inside the type of the accessor placeholder.
           For a usage example check the examples folder
        */
        template < int_t... Args >
        using set = accessor_mixed< AccessorType, pair_< Inxs, Args >... >;
    };

    template < typename... Types >
    struct is_accessor< accessor_mixed< Types... > > : boost::mpl::true_ {};

    template < typename... Types >
    struct is_grid_accessor< accessor_mixed< Types... > > : boost::mpl::true_ {};

    template < typename Accessor, typename ArgsMap, typename... Pairs >
    struct remap_accessor_type< accessor_mixed< Accessor, Pairs... >, ArgsMap > {
        typedef accessor_mixed< typename remap_accessor_type< Accessor, ArgsMap >::type, Pairs... > type;
    };
}
