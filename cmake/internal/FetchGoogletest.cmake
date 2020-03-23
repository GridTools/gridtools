function(fetch_googletest)
    # include Threads manually before googletest such that we can properly apply the workaround
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package( Threads REQUIRED )
    include(workaround_threads)
    _fix_threads_flags()

    # The gtest library needs to be built as static library to avoid RPATH issues
    set(BUILD_SHARED_LIBS OFF)

    include(FetchContent)
    set(INSTALL_GTEST OFF)
    cmake_policy(SET CMP0077 NEW)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        release-1.8.1
    )
    FetchContent_MakeAvailable(googletest)
endfunction()
