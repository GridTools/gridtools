#pragma once


namespace gridtools {
    /**
       Enum defining the directions in a discrete Cartesian grid
     */
    enum sign {any=-2, minus=-1, zero, plus};

    /**
       Class defining a direction in a cartesian 3D grid.
     */
    template <sign I_, sign J_, sign K_>
    struct direction {
        static const sign I = I_;
        static const sign J = J_;
        static const sign K = K_;
    };

    template <sign I, sign J, sign K>
    std::ostream & operator<<(std::ostream& s, direction<I,J,K> const &) {
        s << "dierction<" << I
                  << ", " << J
                  << ", " << K
                  << ">";
        return s;
    }

} // namespace gridtools
