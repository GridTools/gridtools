#pragma once

#include <boundary-conditions/predicate.hpp>
#include <boundary-conditions/direction.hpp>

struct ij_predicate {
    template < gridtools::sign I, gridtools::sign J >
    bool operator()(gridtools::direction< I, J, gridtools::minus_ >) const {
        return false;
    }
    template < gridtools::sign I, gridtools::sign J >
    bool operator()(gridtools::direction< I, J, gridtools::plus_ >) const {
        return false;
    }
    template < gridtools::sign I, gridtools::sign J >
    bool operator()(gridtools::direction< I, J, gridtools::zero_ >) const {
        return true;
    }
};
