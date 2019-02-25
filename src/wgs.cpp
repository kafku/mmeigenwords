#include<iostream>
#include<pybind11/iostream.h>
#include<pybind11/pybind11.h>
#include<pybind11/eigen.h>
#include<Eigen/Core>
#include<Eigen/Dense>

namespace py=pybind11;

void wgs_core_inplace(Eigen::Ref<Eigen::MatrixXd> Y, const Eigen::Ref<const Eigen::MatrixXd> WY) {
  using Scalar = double;
  using Index = int;
  static const Scalar EPS(1E-6);

  Y.col(0) /= std::sqrt(Y.col(0).transpose() * WY.col(0));
  for(Index i = 1; i < Y.cols(); ++i)
  {
    Y.col(i) -= Y.leftCols(i)
      * (Y.leftCols(i).transpose() * WY.col(i)).eval();

    const Scalar norm = Y.col(i).transpose() * WY.col(i);

    if(norm < EPS)
    {
      Y.rightCols(Y.cols() - i).setZero();
      return;
    }
    Y.col(i) /= std::sqrt(norm);
  }
}

void simd_in_use() {
  std::cout << Eigen::SimdInstructionSetsInUse() << std::endl;
}

PYBIND11_MODULE(wgs, mod) {
  mod.doc() = "core of W-inner Gram-Schmidt";
  mod.def("wgs_core_inplace", &wgs_core_inplace);
  mod.def("simd_in_use", &simd_in_use,
      py::call_guard<py::scoped_ostream_redirect,
                     py::scoped_estream_redirect>());
}
