#pragma once

// IS_SM100 / IS_SM90 compile-time arch macros used by the vendored kernel body.
// Main uses runtime Arch::is_sm100f() instead, so these aren't in main's utils.h.

// For development, we define both IS_SM100 and IS_SM90 when using CLion or
// VSCode IDEs so code highlighting will be correct.
#if defined(__CLION_IDE__) || defined(__VSCODE_IDE__)
#define IS_SM100 1
#define IS_SM90 1
#else
// Detect the CUDA architecture to enable/disable arch-specific kernel paths.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000)
#define IS_SM100 1
#else
#define IS_SM100 0
#endif

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 900)
#define IS_SM90 1
#else
#define IS_SM90 0
#endif
#endif  // defined(__CLION_IDE__) || defined(__VSCODE_IDE__)
