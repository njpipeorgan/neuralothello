#pragma once

#include <cstdint>
#include <vector>
#include <torch/torch.h>

namespace nn
{

auto load_tensor(std::string filename, const std::vector<int64_t> sizes)
{
    auto size = std::accumulate(sizes.begin(), sizes.end(),
        int64_t(1), std::multiplies<>{});
    auto tensor = torch::from_file(filename, false, size);
    tensor = tensor.reshape(c10::IntArrayRef(sizes.data(), sizes.size()));
    return tensor;
}

struct ResBlock : torch::nn::Module
{
    torch::nn::Conv2d      conv1{nullptr};
    torch::nn::BatchNorm2d bn1  {nullptr};
    torch::nn::Conv2d      conv2{nullptr};
    torch::nn::BatchNorm2d bn2  {nullptr};

    ResBlock(int64_t channels, std::string file)
    {
        auto options = torch::nn::Conv2dOptions(channels, channels, {3, 3}).padding({1, 1});
        conv1 = register_module("conv1", torch::nn::Conv2d(options));
        conv2 = register_module("conv2", torch::nn::Conv2d(options));
        auto bn_options = torch::nn::BatchNormOptions(channels).affine(true).track_running_stats(true).eps(0.001);
        bn1 = register_module("bn1", torch::nn::BatchNorm2d(bn_options));
        bn2 = register_module("bn2", torch::nn::BatchNorm2d(bn_options));
        conv1->weight.copy_(load_tensor(file + ".conv1.weight", {channels, channels, 3, 3}));
        conv1->bias.copy_(load_tensor(file + ".conv1.bias", {channels}));
        conv2->weight.copy_(load_tensor(file + ".conv2.weight", {channels, channels, 3, 3}));
        conv2->bias.copy_(load_tensor(file + ".conv2.bias", {channels}));
        bn1->weight.copy_(load_tensor(file + ".bn1.weight", {channels}));
        bn1->bias.copy_(load_tensor(file + ".bn1.bias", {channels}));
        bn1->running_mean.copy_(load_tensor(file + ".bn1.running_mean", {channels}));
        bn1->running_var.copy_(load_tensor(file + ".bn1.running_var", {channels}));
        bn2->weight.copy_(load_tensor(file + ".bn2.weight", {channels}));
        bn2->bias.copy_(load_tensor(file + ".bn2.bias", {channels}));
        bn2->running_mean.copy_(load_tensor(file + ".bn2.running_mean", {channels}));
        bn2->running_var.copy_(load_tensor(file + ".bn2.running_var", {channels}));
        this->eval();
    }

    auto forward(torch::Tensor x)
    {
        auto y = x;
        y = conv1->forward(y);
        y = bn1->forward(y);
        y = torch::relu(y);
        y = conv2->forward(y);
        y = bn2->forward(y);
        x = torch::add(x, y);
        x = torch::relu(x);
        return x;
    }

};

struct InputBlock : torch::nn::Module
{
    torch::nn::Conv2d conv1{nullptr};

    InputBlock(int64_t channels, std::string file)
    {
        auto options = torch::nn::Conv2dOptions(1, channels, {1, 1});
        conv1 = register_module("conv1", torch::nn::Conv2d(options));
        conv1->weight.copy_(load_tensor(file + ".conv1.weight", {channels, 1, 1, 1}));
        conv1->bias.copy_(load_tensor(file + ".conv1.bias", {channels}));
        this->eval();
    }

    auto forward(torch::Tensor x)
    {
        x = x.reshape({x.size(0), 1, 8, 8});
        x = conv1->forward(x);
        x = torch::relu(x);
        return x;
    }
};

struct ValueHead : torch::nn::Module
{
    torch::nn::Conv2d conv1{nullptr};
    torch::nn::Linear fc1{nullptr};
    torch::nn::Linear fc2{nullptr};

    ValueHead(int64_t conv1_in, int64_t conv1_out, int64_t fc1_out, std::string file)
    {
        auto options = torch::nn::Conv2dOptions(conv1_in, conv1_out, {1, 1});
        conv1 = register_module("conv1", torch::nn::Conv2d(options));
        fc1   = register_module("fc1", torch::nn::Linear(conv1_out * 64, fc1_out));
        fc2   = register_module("fc2", torch::nn::Linear(fc1_out, 1));
        conv1->weight.copy_(load_tensor(file + ".conv1.weight", {conv1_out, conv1_in, 1, 1}));
        conv1->bias.copy_(load_tensor(file + ".conv1.bias", {conv1_out}));
        fc1->weight.copy_(load_tensor(file + ".fc1.weight", {fc1_out, conv1_out * 64}));
        fc1->bias.copy_(load_tensor(file + ".fc1.bias", {fc1_out}));
        fc2->weight.copy_(load_tensor(file + ".fc2.weight", {1, fc1_out}));
        fc2->bias.copy_(load_tensor(file + ".fc2.bias", {1}));
        this->eval();
    }

    auto forward(torch::Tensor x)
    {
        x = conv1->forward(x);
        x = torch::flatten(x, 1);
        x = torch::tanh(x);
        x = fc1->forward(x);
        x = torch::tanh(x);
        x = fc2->forward(x);
        x = torch::tanh(x);
        return x;
    }

};

struct PolicyHead : torch::nn::Module
{
    torch::nn::Conv2d conv1{nullptr};
    torch::nn::Linear fc1{nullptr};

    PolicyHead(int64_t conv1_in, int64_t conv1_out, std::string file)
    {
        auto options = torch::nn::Conv2dOptions(conv1_in, conv1_out, {1, 1});
        conv1 = register_module("conv1", torch::nn::Conv2d(options));
        fc1   = register_module("fc1", torch::nn::Linear(conv1_out * 64, 64));
        conv1->weight.copy_(load_tensor(file + ".conv1.weight", {conv1_out, conv1_in, 1, 1}));
        conv1->bias.copy_(load_tensor(file + ".conv1.bias", {conv1_out}));
        fc1->weight.copy_(load_tensor(file + ".fc1.weight", {64, conv1_out * 64}));
        fc1->bias.copy_(load_tensor(file + ".fc1.bias", {64}));
        this->eval();
    }

    auto forward(torch::Tensor x)
    {
        x = conv1->forward(x);
        x = torch::flatten(x, 1);
        x = torch::relu(x);
        x = fc1->forward(x);
        //x = torch::softmax(x, 1);
        return x;
    }
};

struct Net
{
    static constexpr int64_t num_filters = 32;
    InputBlock input;
    ResBlock   res1;
    ResBlock   res2;
    ResBlock   res3;
    ResBlock   res4;
    ValueHead  value;
    PolicyHead policy;

    Net(std::string filename) :
        input(num_filters, filename + "/input"),
        res1(num_filters, filename + "/res1"),
        res2(num_filters, filename + "/res2"),
        res3(num_filters, filename + "/res3"),
        res4(num_filters, filename + "/res4"),
        value(num_filters, 2, 64, filename + "/value"),
        policy(num_filters, 4, filename + "/policy")
    {
    }

    void to_gpu()
    {
        input.to(torch::kCUDA);
        res1.to(torch::kCUDA);
        res2.to(torch::kCUDA);
        res3.to(torch::kCUDA);
        res4.to(torch::kCUDA);
        value.to(torch::kCUDA);
        policy.to(torch::kCUDA);
    }

    void predict(float* boards, size_t num_boards, float* values, float* policies)
    {
        torch::NoGradGuard no_grad;
        auto x = torch::zeros({int64_t(num_boards), 64});
        std::memcpy(x.data_ptr(), boards, sizeof(float) * num_boards * 64);
        auto dev = global_options.target_device == TargetDevice::CPU ? torch::kCPU : torch::kCUDA;
        x = x.to(dev);
        //std::cout << x.to(torch::kCPU) << std::endl;
        x = input.forward(x);
        x = res1.forward(x);
        x = res2.forward(x);
        x = res3.forward(x);
        x = res4.forward(x);
        auto value_out = value.forward(x);
        auto policy_out = policy.forward(x);
        value_out = value_out.to(torch::kCPU);
        policy_out = policy_out.to(torch::kCPU);
        //std::cout << value_out << std::endl;
        //std::cout << policy_out << std::endl;
        std::memcpy(values, value_out.data_ptr(), sizeof(float) * num_boards);
        std::memcpy(policies, policy_out.data_ptr(), sizeof(float) * num_boards * 64);
    }

};

}
