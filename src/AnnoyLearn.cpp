#include "../include/AnnoyLearn.hpp"
#include "../include/pybind11/pybind11.h"
#include "../include/pybind11/stl.h"

namespace py = pybind11;

PYBIND11_MODULE(AnnoyLearn, m)
{
  // define all classes
  py::class_<AnnoyBatchLearnWorker>(m, "VQLearn")
    // Constructor 
    .def(py::init<int, std::vector<double>, std::vector<double>, int, double, double, double, double>(), 
    pybind11::arg("d"), pybind11::arg("X"), pybind11::arg("W"), pybind11::arg("n_epochs"), 
    pybind11::arg("rho0")=double(-1.0), pybind11::arg("rho_anneal")=0.95, pybind11::arg("rho_min")=double(0.75), pybind11::arg("min_h") = 0.01) 

    // Methods 
    .def("train", &AnnoyBatchLearnWorker::train)
    // Attributes, set during construction 
    .def_readonly("d", &AnnoyBatchLearnWorker::d)
    .def_readonly("X", &AnnoyBatchLearnWorker::X)
    .def_readonly("W", &AnnoyBatchLearnWorker::W)
    .def_readonly("n_epochs", &AnnoyBatchLearnWorker::n_epochs)
    
    .def_readonly("rho0", &AnnoyBatchLearnWorker::rho0)
    .def_readonly("rho_anneal", &AnnoyBatchLearnWorker::rho_anneal)
    .def_readonly("rho_min", &AnnoyBatchLearnWorker::rho_min)
    
    .def_readonly("min_h", &AnnoyBatchLearnWorker::min_h)

    .def_readonly("N", &AnnoyBatchLearnWorker::N)
    .def_readonly("M", &AnnoyBatchLearnWorker::M);
}
