#include "interface/logging.h"
#include <boost/format.hpp>
#include <chrono>

namespace gridtools {

    namespace _impl {

        /// Get current date-time (up to ms accuracy)
        static std::string getCurrentTime() {
            std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
            auto now_ms = now.time_since_epoch();
            auto now_sec = std::chrono::duration_cast< std::chrono::seconds >(now_ms);
            auto tm_ms = std::chrono::duration_cast< std::chrono::milliseconds >(now_ms - now_sec);

            std::time_t currentTime = std::chrono::system_clock::to_time_t(now);
            struct tm *localTime = std::localtime(&currentTime);

            return (boost::format("%02i:%02i:%02i.%03i") % localTime->tm_hour % localTime->tm_min % localTime->tm_sec %
                       tm_ms.count())
                .str();
        }

        NullLogger &NullLogger::getInstance() noexcept {
            static NullLogger instance_;
            return instance_;
        }

        Logger &Logger::getInstance() noexcept {
            static Logger instance_;
            return instance_;
        }

        LoggerProxy::~LoggerProxy() { std::cout << std::endl; }

        void Logger::print_common() {
            std::cout << "[" << getCurrentTime() << "] ";
            std::cout << std::string(nesting_.size(), ' ');
        }

        LoggerProxy Logger::trace() noexcept {
            print_common();
            std::cout << "[trace] ";
            return LoggerProxy();
        }

        LoggerProxy Logger::debug() noexcept {
            print_common();
            std::cout << "[debug] ";
            return LoggerProxy();
        }

        LoggerProxy Logger::info() noexcept {
            print_common();
            std::cout << "[info] ";
            return LoggerProxy();
        }

        LoggerProxy Logger::warning() noexcept {
            print_common();
            std::cout << "[warning] ";
            return LoggerProxy();
        }

        LoggerProxy Logger::error() noexcept {
            print_common();
            std::cout << "[error] ";
            return LoggerProxy();
        }

        LoggerProxy Logger::fatal() noexcept {
            print_common();
            std::cout << "[fatal] ";
            return LoggerProxy();
        }

        void Logger::push(const std::string str) noexcept {
            print_common();
            std::cout << "[begin] " << str << std::endl;
            nesting_.push_back(str);
        }

        void Logger::pop() noexcept {
            std::string msg = nesting_.back();
            nesting_.pop_back();
            print_common();
            std::cout << "[end  ] " << msg << std::endl;
        }

        bool LoggingIsEnabled = false;

    } // namespace _impl
}
