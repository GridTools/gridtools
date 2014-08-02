#pragma once

#ifndef _IS_TEMPORARY_STORAGE_H_
#define _IS_TEMPORARY_STORAGE_H_

namespace gridtools {
    template <typename T>
    struct is_temporary_storage {
        typedef boost::false_type type;
    };
} // namespace gridtools

#endif
