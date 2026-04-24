#pragma once

#include <span>

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <kerutils/supplemental/torch_tensors.h>

#include <cutlass/bfloat16.h>

static constexpr float LOG_2_E = 1.44269504f;

// Instantiation for tensor.data_ptr<cutlass::bfloat16_t>()
template<>
inline cutlass::bfloat16_t* at::TensorBase::data_ptr<cutlass::bfloat16_t>() const {
    return reinterpret_cast<cutlass::bfloat16_t*>(this->data_ptr());
}

// A struct that holds the architecture information of the current GPU.
struct Arch {
    int major;
    int minor;
    int num_sms;
    cudaDeviceProp* device_prop;

    Arch() {
        device_prop = at::cuda::getCurrentDeviceProperties();
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
        TORCH_CHECK(false, "[FlashMLA] Stride exceeds int32 limit: ", orig_stride);
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
            TORCH_CHECK(false, "Unsupported num_heads_q: ", NUM_HEADS); \
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
        TORCH_CHECK(false, "Unsupported head_dim_qk: ", HEAD_DIM); \
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
        TORCH_CHECK(false, "Unsupported model type: ", (int)MODEL_TYPE); \
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
            TORCH_CHECK(false, "The chosen implementation does not support all required features. See message above for details.");
        }
    }

    inline void run(const RunArgT &params, const std::vector<FeatureT> &required_features) {
        check_if_all_features_are_supported_and_abort(required_features);
        run_(params, required_features);
    }
};

