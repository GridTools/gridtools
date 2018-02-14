#pragma once
#include <memory>
#include <vector>
#include <iostream>

namespace gridtools {
    namespace _impl {

        class NullLogger {
          public:
            static NullLogger &getInstance() noexcept;

            template < class T >
            NullLogger &operator<<(T &&t) noexcept {
                return (*this);
            }
        };

        class LoggerProxy {
          public:
            ~LoggerProxy();

            template < class StreamableValueType >
            LoggerProxy &operator<<(StreamableValueType &&value) {
                std::cout << value;
                return *this;
            }
        };

        class Logger {
          public:
            Logger() = default;

            static Logger &getInstance() noexcept;

            LoggerProxy trace() noexcept;
            LoggerProxy debug() noexcept;
            LoggerProxy info() noexcept;
            LoggerProxy warning() noexcept;
            LoggerProxy error() noexcept;
            LoggerProxy fatal() noexcept;

            void push(const std::string) noexcept;
            void pop() noexcept;

          private:
            std::vector< std::string > nesting_;
            void print_common();
        };

        extern bool LoggingIsEnabled;

    } // namespace _impl

    /// \brief Control the logging behaviour
    ///
    /// For logging use the macro LOG(severity)
    class Logging {
      public:
        Logging() = delete;

        /// \brief Disable logging
        static void enable() noexcept { _impl::LoggingIsEnabled = true; }

        /// \brief Enable logging
        static void disable() noexcept { _impl::LoggingIsEnabled = false; }

        /// \brief Return true if logging is eneabled
        static bool isEnabled() noexcept { return _impl::LoggingIsEnabled; }
    };

/// \macro LOG
/// \brief Logging infrastructure
///
/// The macro is used to initiate logging. The `lvl` argument of the macro specifies one of the
/// following severity levels: `trace`, `debug`, `info`, `warning`, `error` or `fatal`.
///
/// The logging can be completely turned off by defining `GRIDTOOLS_DISABLE_LOGGING`.
///
/// \code
///   LOG(info) << "Hello, world!";
/// \endcode
///
#define LOG(severity) GRIDTOOLS_INTERNAL_LOG(severity)
#define LOG_BEGIN(str) GRIDTOOLS_INTERNAL_LOG_BEGIN(str)
#define LOG_END() GRIDTOOLS_INTERNAL_LOG_END

#ifdef GRIDTOOLS_DISABLE_LOGGING

#define GRIDTOOLS_INTERNAL_LOG(severity) \
    while (0)                            \
    gridtools::_impl::NullLogger::getInstance()

#define GRIDTOOLS_INTERNAL_LOG_BEGIN(str)
#define GRIDTOOLS_INTERNAL_LOG_END
#else

#define GRIDTOOLS_INTERNAL_LOG(severity) \
    if (gridtools::Logging::isEnabled()) \
    gridtools::_impl::Logger::getInstance().severity()

#define GRIDTOOLS_INTERNAL_LOG_BEGIN(str) \
    if (gridtools::Logging::isEnabled())  \
        gridtools::_impl::Logger::getInstance().push(str);

#define GRIDTOOLS_INTERNAL_LOG_END       \
    if (gridtools::Logging::isEnabled()) \
        gridtools::_impl::Logger::getInstance().pop();

#endif
}
