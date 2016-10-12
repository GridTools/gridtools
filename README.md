gridtools
=========

Design project for a C++ library for applications on regular or block regular grids, like PDE solvers.

Coding conventions
==================

the coding conventions follow those implemented by Boost.
* The object member names are lowercase, start with 'm_' and each word is split with an underscore (m_class_member)
* The static object member names are lowercase, start with 's_' and each word is split with an underscore (s_static_member)
* the object names are lowercase and and each word is split with an underscore (class_name)
* the template arguments have the first letter of each word capitalized, no underscores (TemplateArgument)
* the local variables are lowercase and the words are separated by underscores (same as for object names, well, it's not easy mistake them)
* the typedef names are lowercase, words separated by underscores, and they end with _t (some_def_t)
* the class methods and all the functions follow the same convention: lowercase, words separated by underscores
* enum types follow the convention of the object names, but are defined inside the enumtype namespace (enumtype::some_id)

Unit test naming conventions
==================
* the unit test folder structure has to match the code folder structure (e.g., a unit test for some element in `gridtools/include/storage/` has to be placed in `gridtools/unit_tests/storage/`)
* the unit test file name has to match following regex ```/(test)(_cxx11)?([a-z|A-Z|0-9|_]*)((\.cpp|\.cu))```
* cuda unit test file extension is `.cu`. host unit test file extension is `.cpp`
* as described in the regular expression cxx11 tests have to be named like `test_cxx11_<testname>.<cpp|cu>`
* the cmakelists mustn't be modified. tests are discovered automatically if named correctly.
* if you have a very special test case put it into examples or in case it really has to be in the unit_test folder discuss with the gridtools members how to proceed.
