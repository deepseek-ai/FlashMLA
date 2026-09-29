#pragma once
/*
When bisheng compiles `.asc` files with `-std=c++20`, a bunch of header files cannot be included, including `<string>`, because of wrong system-level library being used. So we provide a minimal version of kerutils utilities for `.asc` files to use here.
*/

#include "kerutils/common/common.h"

#include <malloc.h>

#include <cstdio>
#include <exception>

namespace kerutils {

class KUException_NPU final : public std::exception {
    static constexpr uint32_t MAX_MESSAGE_LEN = 2048;
    char* message;

public:
    template<typename... Args>
    explicit KUException_NPU(const char *name, const char* file, const int line, const char* error) {
        message = (char*)malloc(MAX_MESSAGE_LEN);
        if (message == nullptr) {
            fprintf(stderr, "Failed to malloc error message!\n");
            return;
        }

        snprintf(message, MAX_MESSAGE_LEN, "%s error (%s: %d): %s", name, file, line, error);   
    }

    const char *what() const noexcept override {
        return message;
    }
};

#define THROW_KU_EXCEPTION_NPU(name, ...) \
    throw kerutils::KUException_NPU(name, __FILE__, __LINE__, __VA_ARGS__)

// This `KU_ASSERT_NPU` is triggered no matter if the code is compiled with `-DNDEBUG` or not.
#define KU_ASSERT_NPU(cond, ...)                                                                     \
    do {                                                                                         \
        if (not (cond)) {                                                                        \
            char _ku_buf[1024];                                                                  \
            int _ku_len = snprintf(_ku_buf, sizeof(_ku_buf),                                    \
                                   "Assertion `%s` failed (%s:%d)", #cond, __FILE__, __LINE__); \
            __VA_OPT__(                                                                          \
                _ku_len += snprintf(_ku_buf + _ku_len, sizeof(_ku_buf) - _ku_len,              \
                                    ": " __VA_ARGS__);                                          \
            )                                                                                    \
            fprintf(stderr, "%s\n", _ku_buf);                                                    \
            THROW_KU_EXCEPTION_NPU("Assertion", _ku_buf);                                            \
        }                                                                                        \
    } while(0)

#define KU_ACLRT_CHECK_NPU(call)                                                                                   \
    do {                                                                                                            \
        aclError status_ = (call);                                                                                  \
        if (status_ != ACL_SUCCESS) {                                                                               \
            char _ku_buf[1024];                                                                                     \
            snprintf(_ku_buf, sizeof(_ku_buf), "ACLRT error (%s:%d): %s (code %d)", __FILE__, __LINE__,            \
                    aclGetRecentErrMsg(), status_);                                                                \
            fprintf(stderr, "%s\n", _ku_buf);                                                                       \
            THROW_KU_EXCEPTION_NPU("ACLRT", _ku_buf);                                                                   \
        }                                                                                                            \
    } while(0)

}