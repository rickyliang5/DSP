/*
 * Copyright (C) 2025 Linux Studio Plugins Project <https://lsp-plug.in/>
 *           (C) 2025 Vladimir Sadovnikov <sadko4u@gmail.com>
 *
 * This file is part of lsp-dsp-lib
 * Created on: 31 мар. 2020 г.
 *
 * lsp-dsp-lib is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * any later version.
 *
 * lsp-dsp-lib is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with lsp-dsp-lib. If not, see <https://www.gnu.org/licenses/>.
 */

#include <lsp-plug.in/common/alloc.h>
#include <lsp-plug.in/common/types.h>
#include <lsp-plug.in/dsp/dsp.h>
#include <lsp-plug.in/test-fw/helpers.h>
#include <lsp-plug.in/test-fw/ptest.h>

#define MIN_RANK 8
#define MAX_RANK 16

namespace lsp
{
    namespace generic
    {
        void direct_fft(float *dst_re, float *dst_im, const float *src_re, const float *src_im, size_t rank);
    }

    IF_ARCH_X86(
        namespace sse
        {
            void direct_fft(float *dst_re, float *dst_im, const float *src_re, const float *src_im, size_t rank);
        }

        namespace avx
        {
            void direct_fft(float *dst_re, float *dst_im, const float *src_re, const float *src_im, size_t rank);
            void direct_fft_fma3(float *dst_re, float *dst_im, const float *src_re, const float *src_im, size_t rank);
        }

        namespace avx512
        {
            void direct_fft(float *dst_re, float *dst_im, const float *src_re, const float *src_im, size_t rank);
        }
    )

    IF_ARCH_ARM(
        namespace neon_d32
        {
            void direct_fft(float *dst_re, float *dst_im, const float *src_re, const float *src_im, size_t rank);
        }
    )

    IF_ARCH_AARCH64(
        namespace asimd
        {
            void direct_fft(float *dst_re, float *dst_im, const float *src_re, const float *src_im, size_t rank);
        }
    )

    typedef void (* direct_fft_t) (float *dst_re, float *dst_im, const float *src_re, const float *src_im, size_t rank);
}

//-----------------------------------------------------------------------------
// Performance test for complex multiplication
PTEST_BEGIN("dsp.fft", fft, 10, 1000)

    void call(const char *label, float *fft_re, float *fft_im, const float *sig_re, const float *sig_im, size_t rank, direct_fft_t fft)
    {
        if (!PTEST_SUPPORTED(fft))
            return;

        char buf[80];
        snprintf(buf, sizeof(buf), "%s x %d", label, int(1 << rank));
        printf("Testing %s samples (rank = %d) ...\n", buf, int(rank));

        PTEST_LOOP(buf,
            fft(fft_re, fft_im, sig_re, sig_im, rank);
        )
    }

    PTEST_MAIN
    {
        size_t fft_size = 1 << MAX_RANK;

        uint8_t *data   = NULL;

        float *sig_re   = alloc_aligned<float>(data, fft_size * 4, 64);
        float *sig_im   = &sig_re[fft_size];
        float *fft_re   = &sig_im[fft_size];
        float *fft_im   = &fft_re[fft_size];

        for (size_t i=0; i < (1 << MAX_RANK); ++i)
        {
            sig_re[i]       = randf(0.0f, 1.0f);
            sig_im[i]       = 0.0f;
        }

        #define CALL(func) \
            call(#func, fft_re, fft_im, sig_re, sig_im, i, func)

        for (size_t i=MIN_RANK; i <= MAX_RANK; ++i)
        {
            CALL(generic::direct_fft);
            IF_ARCH_X86(CALL(sse::direct_fft));
            IF_ARCH_X86(CALL(avx::direct_fft));
            IF_ARCH_X86(CALL(avx::direct_fft_fma3));
            IF_ARCH_X86(CALL(avx512::direct_fft));
            IF_ARCH_ARM(CALL(neon_d32::direct_fft));
            IF_ARCH_AARCH64(CALL(asimd::direct_fft));
            PTEST_SEPARATOR;
        }

        free_aligned(data);
    }
PTEST_END





