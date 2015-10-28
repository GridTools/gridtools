#pragma once

namespace gridtools {
    template <typename T>
    struct is_storage : boost::mpl::false_{};
} // namespace gridtools


namespace gridtools {
    template <typename T>
    struct is_temporary_storage:boost::mpl::false_{};
} // namespace gridtools
