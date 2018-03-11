#include <immintrin.h>
#include <iostream>
#include <jni.h>
#include <mmul.h>


static void mmul_tiled_avx_unrolled(const int n, const float *left, const float *right, float *result) {
    const int block_width = n >= 256 ? 512 : 256;
    const int block_height = n >= 512 ? 8 : n >= 256 ? 16 : 32;
    for (int column_offset = 0; column_offset < n; column_offset += block_width) {
        for (int row_offset = 0; row_offset < n; row_offset += block_height) {
            for (int i = 0; i < n; ++i) {
                for (int j = column_offset; j < column_offset + block_width && j < n; j += 64) {
                    __m256 sum1 = _mm256_load_ps(result + i * n + j);
                    __m256 sum2 = _mm256_load_ps(result + i * n + j + 8);
                    __m256 sum3 = _mm256_load_ps(result + i * n + j + 16);
                    __m256 sum4 = _mm256_load_ps(result + i * n + j + 24);
                    __m256 sum5 = _mm256_load_ps(result + i * n + j + 32);
                    __m256 sum6 = _mm256_load_ps(result + i * n + j + 40);
                    __m256 sum7 = _mm256_load_ps(result + i * n + j + 48);
                    __m256 sum8 = _mm256_load_ps(result + i * n + j + 56);
                    for (int k = row_offset; k < row_offset + block_height && k < n; ++k) {
                        __m256 multiplier = _mm256_set1_ps(left[i * n + k]);
                        sum1 = _mm256_fmadd_ps(multiplier, _mm256_load_ps(right + k * n + j), sum1);
                        sum2 = _mm256_fmadd_ps(multiplier, _mm256_load_ps(right + k * n + j + 8), sum2);
                        sum3 = _mm256_fmadd_ps(multiplier, _mm256_load_ps(right + k * n + j + 16), sum3);
                        sum4 = _mm256_fmadd_ps(multiplier, _mm256_load_ps(right + k * n + j + 24), sum4);
                        sum5 = _mm256_fmadd_ps(multiplier, _mm256_load_ps(right + k * n + j + 32), sum5);
                        sum6 = _mm256_fmadd_ps(multiplier, _mm256_load_ps(right + k * n + j + 40), sum6);
                        sum7 = _mm256_fmadd_ps(multiplier, _mm256_load_ps(right + k * n + j + 48), sum7);
                        sum8 = _mm256_fmadd_ps(multiplier, _mm256_load_ps(right + k * n + j + 56), sum8);
                    }
                    _mm256_store_ps(result + i * n + j, sum1);
                    _mm256_store_ps(result + i * n + j + 8, sum2);
                    _mm256_store_ps(result + i * n + j + 16, sum3);
                    _mm256_store_ps(result + i * n + j + 24, sum4);
                    _mm256_store_ps(result + i * n + j + 32, sum5);
                    _mm256_store_ps(result + i * n + j + 40, sum6);
                    _mm256_store_ps(result + i * n + j + 48, sum7);
                    _mm256_store_ps(result + i * n + j + 56, sum8);
                }
            }
        }
    }
}

JNIEXPORT void JNICALL Java_com_openkappa_mmul_MatrixMultiplication_multiply
  (JNIEnv* env, jclass klass, jobject left, jobject right, jobject result, jint size)
{
	mmul_tiled_avx_unrolled((int) size,
	           (const float*) env->GetDirectBufferAddress(left),
	           (const float*) env->GetDirectBufferAddress(right),
	           (float*) env->GetDirectBufferAddress(result));
}