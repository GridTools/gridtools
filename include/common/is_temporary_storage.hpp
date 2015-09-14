#pragma once

#ifndef _IS_TEMPORARY_STORAGE_H_
#define _IS_TEMPORARY_STORAGE_H_



namespace gridtools {
    template <typename T>
    struct is_storage : boost::mpl::false_{};
} // namespace gridtools


namespace gridtools {
    template <typename T>
    struct is_temporary_storage:boost::mpl::false_{};
} // namespace gridtools

#endif
