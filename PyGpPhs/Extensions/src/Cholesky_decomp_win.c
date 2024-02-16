/*Open commend line, navigate to the directory that has this file
 type in:
 Important: it must be make sure that the c/c++ compiler is the same bit as your machine
 Below is the Microsoft Visual Code tool
 1. cl /EHsc Cholesky_decomp.c

 2. link /DLL /OUT:Cholesky_decomp_64bit.dll Cholesky_decomp_win.obj
 Then the simulation should be able to use
 */

#include <math.h>

__declspec(dllexport)
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
