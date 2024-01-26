/*Open commend line, navigate to the directory that has this file
 type in:
 1. gcc -c -fPIC Cholesky_decomp.c -o Cholesky_decomp.o

 2. gcc -shared -o Cholesky_decomp.so Cholesky_decomp.o
 Then the simulation should be able to use
 */

#include <math.h>

void Cholesky_Decomposition(double* A, double* result, int n) {

    // Decomposing a matrix into Lower Triangular
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            double sum = 0;

            if (j == i) {
                for (int k = 0; k < j; k++)
                    sum += pow(result[j * n + k], 2);
                result[j * n + j] = sqrt(A[j * n + j] - sum);
            } else {
                for (int k = 0; k < j; k++)
                    sum += (result[i * n + k] * result[j * n + k]);
                result[i * n + j] = (A[i * n + j] - sum) / result[j * n + j];
            }
        }
    }
}