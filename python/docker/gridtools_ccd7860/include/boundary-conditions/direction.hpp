#pragma once

/**
@file
@brief definition of direction in a 3D cartesian grid
 */
namespace gridtools {
    /**
       @brief Enum defining the directions in a discrete Cartesian grid
     */
    enum sign {any_=-2, minus_=-1, zero_, plus_};
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
    template <sign I_, sign J_, sign K_, class Predicate=boost::enable_if_c<true>::type >
    struct direction {
        static const sign I = I_;
        static const sign J = J_;
        static const sign K = K_;
    };

    template <sign I, sign J, sign K>
    std::ostream & operator<<(std::ostream& s, direction<I,J,K> const &) {
        s << "direction<" << I
                  << ", " << J
                  << ", " << K
                  << ">";
        return s;
    }

} // namespace gridtools
