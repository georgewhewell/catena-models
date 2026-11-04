
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#ifdef CATENA_EXP_MPFR
#include <mpfr.h>
#endif

static uint32_t bits(float x) { uint32_t b; std::memcpy(&b, &x, 4); return b; }
static float word(uint32_t b) { float x; std::memcpy(&x, &b, 4); return x; }
static void check(hipError_t status) {
    if (status != hipSuccess) { std::fprintf(stderr, "%s\n", hipGetErrorString(status)); std::exit(1); }
}
__global__ void evaluate(const float* input, float* output, size_t count) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) EXP_SYMBOL(input[i], output + i);
}
int main() {
#ifdef CATENA_EXP_MPFR
    mpfr_t oracle;
    mpfr_init2(oracle, 160);
#endif
    std::vector<float> input;
    // Signed zeros, subnormals, infinities, signaling/quiet NaNs, extreme finite values.
    for (uint32_t b : {0u, 1u, 0x007fffffu, 0x00800000u, 0x7f7fffffu,
                       0x7f800000u, 0x7f800001u, 0x7fc12345u}) {
        input.push_back(word(b)); input.push_back(word(b | 0x80000000u));
    }
    // Neighboring floats at overflow, zero/subnormal and subnormal/normal transitions.
    for (uint32_t center : {0x39000000u, 0x3eb17218u, 0x42b17218u, 0x42cff1b5u, 0x42aeac50u})
        for (int i = -1024; i <= 1024; ++i) {
            input.push_back(word(center + i)); input.push_back(-word(center + i));
        }
    for (int k = -150; k <= 128; ++k) {
        float x = float((k + 0.5L) * 0.693147180559945309417232121458176568L);
        uint32_t b = bits(x);
        for (int i = -16; i <= 16; ++i) input.push_back(word(b + i));
    }
    // All BF16 inputs, plus random F32 words and samples in the active exp range.
    for (uint32_t b = 0; b < 65536; ++b) input.push_back(word(b << 16));
    uint32_t rng = 123456789;
    for (int i = 0; i < 1000000; ++i) {
        rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
        input.push_back(word(rng));
        input.push_back(-104.0f + 193.0f * float(rng >> 8) * 0x1p-24f);
    }
    // Optional exhaustive sweep of every nontrivial finite input (about 327M).
    // Tiny inputs are 1+x, and all magnitudes above these limits saturate.
    const bool exhaustive = std::getenv("CATENA_EXP_EXHAUSTIVE") != nullptr;
    uint64_t total = 0, different = 0;
    uint32_t max_ulp = 0;
    auto run = [&](const std::vector<float>& values) {
        float *device_input, *device_output;
        const size_t bytes = values.size() * sizeof(float);
        check(hipMalloc(&device_input, bytes)); check(hipMalloc(&device_output, bytes));
        check(hipMemcpy(device_input, values.data(), bytes, hipMemcpyHostToDevice));
        evaluate<<<(values.size() + 255) / 256, 256>>>(device_input, device_output, values.size());
        check(hipGetLastError());
        std::vector<float> got(values.size());
        check(hipMemcpy(got.data(), device_output, bytes, hipMemcpyDeviceToHost));
        for (size_t i = 0; i < values.size(); ++i) {
            const float x = values[i];
            float host; EXP_SYMBOL(x, &host);
            uint32_t expected;
            if (std::isnan(x)) expected = 0x7fc00000u;
            else {
#ifdef CATENA_EXP_MPFR
                mpfr_set_flt(oracle, x, MPFR_RNDN);
                mpfr_exp(oracle, oracle, MPFR_RNDN);
                expected = bits(mpfr_get_flt(oracle, MPFR_RNDN));
#else
                expected = bits(float(std::exp((long double)x)));
#endif
            }
            uint32_t actual = bits(got[i]);
            uint32_t ulp = actual > expected ? actual - expected : expected - actual;
            max_ulp = std::max(max_ulp, ulp);
            different += actual != expected;
            const bool exact = !std::isfinite(x) || x == 0.0f
                || x >= word(0x42b17218u) || x <= word(0xc2cff1b5u);
            if (actual != bits(host) || ulp > 1 || (exact && actual != expected)) {
                std::fprintf(stderr, "input=%08x host=%08x gpu=%08x reference=%08x ulp=%u\n",
                             bits(x), bits(host), actual, expected, ulp);
                std::exit(1);
            }
        }
        total += values.size();
        check(hipFree(device_input)); check(hipFree(device_output));
    };
    run(input);
    if (exhaustive) {
        input.clear();
        for (uint32_t b = 0x39000000u; b < 0x42cff1b5u; ++b) {
            input.push_back(-word(b));
            if (b < 0x42b17218u) input.push_back(word(b));
            if (input.size() >= 1048576) { run(input); input.clear(); }
        }
        if (!input.empty()) run(input);
    }
    std::printf("exp: %llu inputs, host/GPU bits identical, max error %u ULP, %llu differ from ",
                (unsigned long long)total, max_ulp, (unsigned long long)different);
#ifdef CATENA_EXP_MPFR
    std::puts("160-bit MPFR oracle");
    mpfr_clear(oracle);
#else
    std::puts("long-double oracle");
#endif
}
