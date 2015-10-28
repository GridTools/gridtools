/*
* Copyright (c) 2014, Carlos Osuna.
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*     * Redistributions of source code must retain the above copyright
*       notice, this list of conditions and the following disclaimer.
*     * Redistributions in binary form must reproduce the above copyright
*       notice, this list of conditions and the following disclaimer in the
*       documentation and/or other materials provided with the distribution.
*     * Neither the name of the organization nor the
*       names of its contributors may be used to endorse or promote products
*       derived from this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
* ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL CARLOS OSUNA BE LIABLE FOR ANY
* DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
* ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
// Extracted from Andrei Alexandrescu @GoingNative2013
#pragma once
#include "common/array.hpp"

#ifdef CXX11_ENABLED

namespace gridtools {

template <unsigned K, class R, class F, class Array>
struct Expander{
	template< class... Us>
    static R expand(Array&& a, Us&&... args) {
        return Expander<K-1, R, F, Array>::expand(
                a,
                a[K-1],
                args...);
	}
};

template<class F, class R, class Array>
struct Expander<0, R, F, Array> {
	template<class... Us>
    static R expand(Array&&, Us...args){
        return F::apply (args...);
	}
};

template <unsigned K, class R, class F, typename Inj, class Array>
struct Expander_inj{
    template<class... Us>
    static R expand(const Inj& inj, Array&& a, Us&&... args) {
        return Expander_inj<K-1, R, F, Inj, Array>::expand(
                inj,
                a,
                a[K-1],
                args...);
    }
};

template<class R, class F, typename Inj, class Array>
struct Expander_inj<0, R, F, Inj, Array> {
    template<class... Us>
    static R expand(const Inj& inj, Array&&, Us...args){
        return F::apply (inj, args...);
    }
};

template <typename ReturnType, typename Fn, typename Array>
static auto explode(const Array & a)
-> ReturnType
{
        GRIDTOOLS_STATIC_ASSERT((is_array<Array>::value), "Error: Wrong Type");
        return Expander<Array::n_dimensions,
            ReturnType,
            Fn,
            const Array& >::expand(a);
}

template <typename ReturnType, typename Fn, typename Array, typename Inj>
static auto explode(const Array & a, const Inj& inj)
-> ReturnType
{
        GRIDTOOLS_STATIC_ASSERT((is_array<Array>::value), "Error: Wrong Type");
        return Expander_inj<Array::n_dimensions,
            ReturnType,
            Fn,
            Inj,
            const Array& >::expand(inj, a);
}


}

#endif
