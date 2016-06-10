#pragma once
#ifdef CXX11_ENABLED
#define new_cond conditional< __COUNTER__ >
#define new_switch_variable switch_variable< __COUNTER__, int >
#else
#define new_cond(a, b) conditional< __COUNTER__ > a(b)
#endif
