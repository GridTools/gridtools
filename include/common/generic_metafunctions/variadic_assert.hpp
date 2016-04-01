#pragma once

#ifdef CXX11_ENABLED

template < typename Lambda, typename First >
void variadic_assert(Lambda fn, First first) {
    assert(fn(first));
}

template < typename Lambda, typename First, typename... T >
void variadic_assert(Lambda fn, First first, T... args) {
    assert(fn(first));
    variadic_assert(fn, args...);
}

#endif
