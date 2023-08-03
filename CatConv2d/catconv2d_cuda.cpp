#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

std::vector<torch::Tensor> catconv2d_cuda_forward(
    std::vector<torch::Tensor> input_list,
    torch::Tensor weights,
    torch::Tensor bias,
    bool relu);


std::vector<torch::Tensor> catconv2d_cuda_backward(
    torch::Tensor grad_output,
    torch::Tensor X,
    torch::Tensor weights,
    torch::Tensor bias);

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> catconv2d_forward(
    std::vector<torch::Tensor> input_list,
    torch::Tensor weights,
    torch::Tensor bias,
    bool relu) {
  CHECK_INPUT(input_list[0]);
  CHECK_INPUT(weights);
  CHECK_INPUT(bias);
  
  return catconv2d_cuda_forward(input_list, weights, bias, relu);
}

std::vector<torch::Tensor> catconv2d_backward(
    torch::Tensor grad_output,
    torch::Tensor X,
    torch::Tensor weights,
    torch::Tensor bias) {
  CHECK_INPUT(grad_output);
  CHECK_INPUT(X);
  CHECK_INPUT(weights);
  CHECK_INPUT(bias);

  return catconv2d_cuda_backward(
      grad_output,
      X,
      weights,
      bias);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward",  &catconv2d_forward,  "CatConv2d forward (CUDA)");
  m.def("backward", &catconv2d_backward, "CatConv2d backward (CUDA)");
}
