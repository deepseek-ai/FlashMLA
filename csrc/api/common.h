#pragma once

#include <span>
#include <array>
#include <deque>
#include <mutex>
#include <vector>

#include <cuda_runtime.h>

#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>

#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>
#include <torch/headeronly/util/shim_utils.h>

#include <kerutils/supplemental/torch_tensors.h>
#include <kerutils/supplemental/cuda_stream.h>

#include <cutlass/bfloat16.h>

using torch::stable::Tensor;
using torch::headeronly::ScalarType;

// Re-exported from kerutils so existing call sites can use it unqualified.
using kerutils::get_current_cuda_stream;

static constexpr float LOG_2_E = 1.44269504f;

// get_cached_device_prop is a helper to access device properties (SM count,
// compute capability) via the CUDA Runtime directly and cache per device index.
// ABI-stable replacement for at::cuda::getCurrentDeviceProperties().
namespace detail {

inline std::deque<std::once_flag> device_flags;
inline std::vector<cudaDeviceProp> device_properties;
inline std::once_flag device_vectors_init_flag;

inline void init_device_vectors() {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    STD_TORCH_CHECK(err == cudaSuccess,
                    "cudaGetDeviceCount failed: ", cudaGetErrorString(err));
    device_flags.resize(device_count);
    device_properties.resize(device_count);
}

inline void init_device_property(int device_index) {
    cudaDeviceProp device_prop{};
    cudaError_t err = cudaGetDeviceProperties(&device_prop, device_index);
    STD_TORCH_CHECK(err == cudaSuccess,
                    "cudaGetDeviceProperties failed: ", cudaGetErrorString(err));
    device_properties[device_index] = device_prop;
}

}  // namespace detail

inline const cudaDeviceProp &get_cached_device_prop() {
    std::call_once(detail::device_vectors_init_flag, detail::init_device_vectors);
    int device_index = static_cast<int>(torch::stable::accelerator::getCurrentDeviceIndex());
    STD_TORCH_CHECK(
        device_index >= 0 &&
            static_cast<size_t>(device_index) < detail::device_properties.size(),
        "CUDA device index ", device_index, " out of range [0, ",
        detail::device_properties.size(), ")");
    std::call_once(detail::device_flags[device_index], detail::init_device_property,
                   device_index);
    return detail::device_properties[device_index];
}

// A struct that holds the architecture information of the current GPU.
struct Arch {
    int major;
    int minor;
    int num_sms;
    const cudaDeviceProp* device_prop;

    Arch() {
        device_prop = &get_cached_device_prop();
        major = device_prop->major;
        minor = device_prop->minor;
        num_sms = device_prop->multiProcessorCount;
    }

    bool is_sm90a() const {
        return major == 9 && minor == 0;
    }

    bool is_sm100f() const {
        return major == 10;
    }
};

// Convert int64_t stride to int32_t, with overflow check.
inline int int64_stride_to_int(int64_t orig_stride) {
    if (orig_stride > std::numeric_limits<int>::max()) {
        STD_TORCH_CHECK(false, "[FlashMLA] Stride exceeds int32 limit: ", orig_stride);
    }
    return static_cast<int>(orig_stride);
}

#define DISPATCH_NUM_HEADS(NUM_HEADS, CONSTEXPR_NAME, ...) \
    [&] () { \
        if (NUM_HEADS == 128) { \
            static constexpr int CONSTEXPR_NAME = 128; \
            return __VA_ARGS__(); \
        } else if (NUM_HEADS == 64) { \
            static constexpr int CONSTEXPR_NAME = 64; \
            return __VA_ARGS__(); \
        } else { \
            STD_TORCH_CHECK(false, "Unsupported num_heads_q: ", NUM_HEADS); \
        } \
    } ();

#define DISPATCH_HEAD_DIM(HEAD_DIM, CONSTEXPR_NAME, ...) \
[&] () { \
    if (HEAD_DIM == 576) { \
        static constexpr int CONSTEXPR_NAME = 576; \
        return __VA_ARGS__(); \
    } else if (HEAD_DIM == 512) { \
        static constexpr int CONSTEXPR_NAME = 512; \
        return __VA_ARGS__(); \
    } else { \
        STD_TORCH_CHECK(false, "Unsupported head_dim_qk: ", HEAD_DIM); \
    } \
} ();

#define DISPATCH_BOOLEAN_FLAG(FLAG, CONSTEXPR_NAME, ...) \
    [&] () { \
        if (FLAG) { \
            static constexpr bool CONSTEXPR_NAME = true; \
            return __VA_ARGS__(); \
        } else { \
            static constexpr bool CONSTEXPR_NAME = false; \
            return __VA_ARGS__(); \
        } \
    } ();

#define DISPATCH_MODEL_TYPE(MODEL_TYPE, CONSTEXPR_NAME, ...) \
[&] () { \
    if (MODEL_TYPE == ModelType::V32) { \
        static constexpr ModelType CONSTEXPR_NAME = ModelType::V32; \
        return __VA_ARGS__(); \
    } else if (MODEL_TYPE == ModelType::MODEL1) { \
        static constexpr ModelType CONSTEXPR_NAME = ModelType::MODEL1; \
        return __VA_ARGS__(); \
    } else { \
        STD_TORCH_CHECK(false, "Unsupported model type: ", (int)MODEL_TYPE); \
    } \
} ();

// The following code is adapted from https://ykiko.me/en/articles/680412313/, which converts enum values to string names.
template<auto value>
constexpr auto get_static_enum_name(){
    std::string_view name;
#if __GNUC__ || __clang__
    name = __PRETTY_FUNCTION__;
    std::size_t start = name.find('=') + 2;
    std::size_t end = name.size() - 1;
    name = std::string_view{ name.data() + start, end - start };
    start = name.find("::");
#elif _MSC_VER
    name = __FUNCSIG__;
    std::size_t start = name.find('<') + 1;
    std::size_t end = name.rfind(">(");
    name = std::string_view{ name.data() + start, end - start };
    start = name.rfind("::");
#endif
    return start == std::string_view::npos ? name : std::string_view {
            name.data() + start + 2, name.size() - start - 2
    };
}

template<typename T, std::size_t N = 0>
static constexpr std::size_t get_enum_max(){
    constexpr T value = static_cast<T>(N);
    if constexpr (get_static_enum_name<value>().find(")") == std::string_view::npos)
        return get_enum_max<T, N + 1>();
    else
        return N;
}

template<typename T> requires std::is_enum_v<T>
static constexpr std::string get_dynamic_enum_name(T value){
    constexpr std::size_t num = get_enum_max<T>();
    constexpr auto names = []<std::size_t... Is>(std::index_sequence<Is...>){
        return std::array<std::string_view, num>{
            get_static_enum_name<static_cast<T>(Is)>()...
        };
    }(std::make_index_sequence<num>{});
    return (std::string)names[static_cast<std::size_t>(value)];
}

// A shortcut macro to declare supported features in an implementation class.
#define DECLARE_SUPPORTED_FEATURES(...) \
protected: \
    static constexpr FeatureT features[] = { __VA_ARGS__ }; \
    constexpr inline std::span<const FeatureT> get_supported_features() const override { \
        return features; \
    }

/*
ImplBase - The base class for every implementation.

Every implementation should inherit from this class and implement the pure virtual functions, including:
- `run_`: The function that runs the implementation.
- `get_supported_features`: The function that returns the supported features of the implementation. You may use `DECLARE_SUPPORTED_FEATURES` to declare the supported features in a concise way.

The dispatcher will invoke `ImplBase::run()`, which checks if all required features are supported by the implementation, and then calls `run_`.
*/
template<
    typename RunArgT_,
    typename FeatureT_
>
class ImplBase {
protected:
    using RunArgT = RunArgT_;
    using FeatureT = FeatureT_;

    virtual inline void run_(const RunArgT &params, const std::vector<FeatureT> &required_features) = 0;

    constexpr virtual inline std::span<const FeatureT> get_supported_features() const = 0;

    virtual ~ImplBase() = default;

public:
    inline bool check_if_all_features_are_supported(const std::vector<FeatureT> &required_features) {
        for (const auto &required_feature : required_features) {
            bool is_supported = false;
            for (const auto &supported_feature : get_supported_features()) {
                if (required_feature == supported_feature) {
                    is_supported = true;
                    break;
                }
            }
            if (!is_supported) {
                return false;
            }
        }
        return true;
    }

    inline void check_if_all_features_are_supported_and_abort(const std::vector<FeatureT> &required_features) {
        if (!check_if_all_features_are_supported(required_features)) {
            fprintf(stderr, "[FlashMLA] Error: The chosen implementation does not support all required features.\n");
            fprintf(stderr, "Required features:\n");
            for (const auto &f : required_features) {
                fprintf(stderr, "  - %3d: %s\n", static_cast<int>(f), get_dynamic_enum_name(f).c_str());
            }
            fprintf(stderr, "\n");
            fprintf(stderr, "Supported features:\n");
            for (const auto &supported_feature : get_supported_features()) {
                fprintf(stderr, "  - %3d: %s\n", static_cast<int>(supported_feature), get_dynamic_enum_name(supported_feature).c_str());
            }
            fprintf(stderr, "\n");
            fprintf(stderr, "Features that are required but not supported:\n");
            for (const auto &required_feature : required_features) {
                bool is_supported = false;
                for (const auto &supported_feature : get_supported_features()) {
                    if (required_feature == supported_feature) {
                        is_supported = true;
                        break;
                    }
                }
                if (!is_supported) {
                    fprintf(stderr, "  - %3d: %s\n", static_cast<int>(required_feature), get_dynamic_enum_name(required_feature).c_str());
                }
            }
            fprintf(stderr, "\n");
            Arch cur_gpu_arch = Arch();
            fprintf(stderr, "Current GPU: %s, SM %d.%d with %d SMs\n", cur_gpu_arch.device_prop->name, cur_gpu_arch.major, cur_gpu_arch.minor, cur_gpu_arch.num_sms);
            fprintf(stderr, "This means that the dispatcher has chosen an implementation that does not support all required features. Maybe there is a bug in the dispatcher, or you have requested an invalid combination of features.\n");
            STD_TORCH_CHECK(false, "The chosen implementation does not support all required features. See message above for details.");
        }
    }

    inline void run(const RunArgT &params, const std::vector<FeatureT> &required_features) {
        check_if_all_features_are_supported_and_abort(required_features);
        run_(params, required_features);
    }
};
