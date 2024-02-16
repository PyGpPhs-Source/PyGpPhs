//PHS_simulation
//Author: Tommy Li
//Date: Dec.12, 2023
/*kernel
 *Kernel - Function for covariance
 To compile and use, use cpp compiler to compile this file to .obj (object file),
 then compile the .obj file to .dll (dynamically linked lib)

 Important: it must be make sure that the c/c++ compiler is the same bit as your machine
 Below is the Microsoft Visual Code tool
 1. cl /EHsc PHSkernel_se_CPP_win.cpp

 2. link /DLL /OUT:kernel_64bit.dll PHSkernel_se_CPP_win.obj

 Then the program should be ready to use
 */
#include "..\Eigen\Dense"
#include <vector>

namespace Eigen{
    typedef Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> MatrixXdRM;
}

/*
 * This is the kernel function
 * @Param:
 *      vec1: the first vector to compute derivative with, Eigen::VectorXd (unknown dim)
 *      vec2: the second vector to compute derivative with, Eigen::VectorXd (unknown dim)
 *      sigma: scalar
 *      l: scalar
 *  @Return:
 *      scalar
 */
Eigen::MatrixXd kernel(const Eigen::MatrixXd &X,
                       const Eigen::MatrixXd &Y,
                       const double sigma,
                       const std::vector<double> &l) {
    uint32_t N = X.rows();
    uint32_t M = Y.rows();
    uint32_t n = X.cols();
    if (n != Y.cols() || l.size() != n) {
        throw std::out_of_range("kernel function::Dimensions of X and Y do not match");
    }
    double sigmaSq = sigma * sigma;
    Eigen::MatrixXd K(N, M);
    for (uint32_t i = 0; i < N; ++i) {
        for (uint32_t j = 0; j < M; ++j) {
            Eigen::VectorXd diff = X.row(i) - Y.row(j);
            Eigen::VectorXd temp(diff);
            for (uint32_t k = 0; k < n; ++k) {
                temp(k) /= (2 * l[k] * l[k]);
            }
            double exponent = temp.transpose() * diff;
            K(i, j) = sigmaSq * exp(-exponent);
        }
    }
    return K.transpose();
}


/*
 * Derivative function for squared exponential kernel
 * @Param:
 *      vec1: the first vector to compute derivative with, Eigen::MatrixXD (unknown dim)
 *      X: could either be matrix or vector. IMPORTANT: have to be col vector
 *      sigma: scalar
 *      l: scalar
 *  @Return:
 *      Eigen::VectorXD
 */
Eigen::MatrixXd Derivative(
        const Eigen::MatrixXd &X,
        const Eigen::MatrixXd &Y,
        const double sigma,
        const std::vector<double> &l
) {
    uint32_t N = X.rows();
    uint32_t M = Y.rows();
    uint32_t n = X.cols();
    if (n != Y.cols() || l.size() != n) {
        throw std::out_of_range("Derivative function::Dimensions of X and Y do not match");
    }
    Eigen::MatrixXd result(N, M * n);
    //Eigen::MatrixXd result(M * n, N);
    for (uint32_t i = 0; i < N; ++i) {
        auto x = X.row(i);
        for (uint32_t j = 0; j < M; ++j) {
            auto xk = Y.row(j);
            for (uint32_t k = 0; k < n; ++k) {
                double part1 = (x(k) - xk(k)) / (l[k] * l[k]);
                auto part2 = kernel(x, xk, sigma, l); //1by1 matrix
                result(i, j * n + k) = part1 * part2(0, 0);
            }
        }
    }
    return result.transpose();

}


/*
 * Compute Partial Derivative (Hessian) for squared exponential kernel
 * @Param:
 *      X: The first matrix-like structure to compute Hessian with, Eigen::MatrixXD (unknown dim; this could
 *          either be an Eigen::Vector or an Eigen::Matrix
 *      Y: The second matrix-like structure to compute Hessian with, Eigen::MatrixXD (unknown dim; this could
 *          either be an Eigen::Vector or an Eigen::Matrix
 *      sigma: scalar
 *      l: scalar
 *  @Return:
 *      Eigen::MatrixXd, the result of the function, regardless of X, Y, will be a Eigen::MatrixXd with dimension
 *          known at runtime.
 */
Eigen::MatrixXd DDerivative(
        const Eigen::MatrixXd &X,
        const Eigen::MatrixXd &Y,
        const double sigma,
        const std::vector<double> &l,
        const Eigen::MatrixXd &JR // should pass identity matrix if do not want to use JR
) {
    // base initialization:
    uint32_t n = X.cols();
    uint32_t N = X.rows();
    uint32_t M = Y.rows();
    if (n != Y.cols() || l.size() != n || JR.rows() != n || JR.cols() != n) {
        throw std::out_of_range("DDerivative function::Dimensions of X and Y do not match");
    }
    Eigen::MatrixXd result(N * n, M * n);

    // X and Y are both row matrix (row vector is datapoint)

    // indexing blocks of hessian matrix between 2 points; R^(nN) x (nM)
    for (uint32_t i = 0; i < N; ++i) {
        auto x = X.row(i);
        for (uint32_t j = 0; j < M; ++j) {
            // now inside the hessian matrix; R^(n) x (n)
            auto y = Y.row(j);
            // JR multiplicant
            Eigen::MatrixXd grid(n, n);
            // The k and m are indexed to only compute the upper triangular matrix due to symmetry
            for (uint32_t k = 0; k < n; ++k) {
                for (uint32_t m = k; m < n; ++m) {
                    auto tmp = kernel(x, y, sigma, l);
                    double kernel_val = tmp(0, 0);
                    double d;
                    if (k == m) {
                        d = std::pow(x(k) - y(m), 2) / std::pow(l[k], 4) - 1 / (l[k] * l[k]);
                        grid(k, m) = -kernel_val * d;
                    } else {
                        d = (x(k) - y(k)) * (x(m) - y(m)) / (l[k] * l[k] * l[m] * l[m]);
                        grid(k, m) = -kernel_val * d;
                        grid(m, k) = -kernel_val * d;
                    }
                    // make sure assignment happens once on diagonal entries (same time complexity)
                }
            }
            grid = JR.transpose() * grid * JR;
            for (uint32_t k = 0; k < n; ++k) {
                for (uint32_t m = 0; m < n; ++m) {
                    result((i) * n + k, (j) * n + m) = grid(k, m);
                }
            }
        }
    }
    return result;
}

extern "C"{
__declspec(dllexport)
void Command_center(double* X_ptr, double* Y_ptr, const double sigma, const double* l_ptr, const int d1, const int rowX, const int colX, const int rowY, const int colY, const int rowL, double* res, double* JR_ptr=nullptr){
    if (colX != colY || colX != rowL){
        throw std::invalid_argument("rows of X, Y, or l do not match!");
    }
    Eigen::MatrixXdRM X(X_ptr, rowX, colX);
    Eigen::MatrixXdRM Y(Y_ptr, rowY, colY);
    auto l = std::vector<double>(l_ptr, l_ptr + rowL);
    Eigen::MatrixXd result;
    uint32_t resLength;
    if (d1 == 0){
        result = kernel(X, Y, sigma, l);
        resLength = rowX * rowY;
    }else if (d1 == 1){
        result = Derivative(X, Y, sigma, l);
        resLength = rowX * (rowY * colX);
    }else if(d1 == 2){
        resLength = rowX * rowY * colX * colX;
        if (JR_ptr == nullptr)
            result = DDerivative(X, Y, sigma, l, Eigen::MatrixXd::Identity(X.cols(), X.cols()));
        else{
            Eigen::MatrixXdRM JR(JR_ptr, colX, colX);
            result = DDerivative(X, Y, sigma, l, JR);
        }
    }else{
        throw std::invalid_argument("invalid action variable d1!");
    }
    auto temp = result.data();
    std::copy(temp, temp + resLength, res);
}
}