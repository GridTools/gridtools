#pragma once

#ifndef _IS_TEMPORARY_STORAGE_H_
#define _IS_TEMPORARY_STORAGE_H_
#endif


namespace gridtools {
    template <typename T>
    struct is_actual_storage : boost::mpl::false_{};

    template <typename T>
    struct is_temporary_storage:boost::mpl::false_{};
} // namespace gridtools
