#pragma once

#include <torch/torch.h>

struct BasicBlockImpl : public torch::nn::Module 
{
    BasicBlockImpl(int input_ch, int outp_ch, int kernel_size, int stride_steps)
    :   convXX(torch::nn::Conv2dOptions(input_ch, outp_ch, kernel_size).stride(stride_steps)),
        bnXX(torch::nn::BatchNorm2d(outp_ch)) {
            register_module("convXX", convXX);
            register_module("bnXX", bnXX);
        }
        torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(torch::max_pool2d(convXX->forward(x), 2));
        x = torch::batch_norm(bnXX->forward(x), bnXXW,bnBiasXXW,bnmeanXXW,bnvarXXW,true,0.9,0.001,true);
    return x;
  }
  torch::nn::Conv2d convXX;
  torch::Tensor bnXXW;
  torch::Tensor bnBiasXXW;
  torch::Tensor bnmeanXXW;
  torch::Tensor bnvarXXW;
  torch::nn::BatchNorm2d bnXX;

};
TORCH_MODULE(BasicBlock);

struct ObscureResNetImpl : public torch::nn::Module 
{
  ObscureResNetImpl(int number_of_classes) 
  :     Block1(3,20,5,2),
        Block2(20,30,5,1),
        Block3(30,30,5,1),
        Block4(30,30,5,1),
        fc1(270, 100),
        fc2(100, number_of_classes) {

    register_module("Block1", Block1);
    register_module("Block2", Block2);
    register_module("Block3", Block3);
    register_module("Block4", Block4);
    register_module("fc1", fc1);
    register_module("fc2", fc2);
  }
    torch::Tensor forward(torch::Tensor x) {

    x = Block1->forward(x);
    x = Block2->forward(x);
    x = Block3->forward(x);
    x = Block4->forward(x);
    
    x = x.view({-1, 270});
    x = torch::relu(fc1->forward(x));
    x = torch::dropout(x, /*p=*/0.5, /*training=*/is_training());
    x = fc2->forward(x);
    return torch::log_softmax(x, /*dim=*/1);
  }

  BasicBlock Block1;
  BasicBlock Block2;
  BasicBlock Block3;
  BasicBlock Block4;

  torch::nn::Linear fc1;
  torch::nn::Linear fc2;
};

TORCH_MODULE(ObscureResNet);