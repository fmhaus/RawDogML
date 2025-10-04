#pragma once

#include "types.h"
#include "tensor.h"
#include "statedict.h"

#include <memory>
#include <vector>
#include <random>
#include <string>

struct Parameter
{
	Parameter(u32 size, const std::string& name);

	void init_normal_dist(std::mt19937& gen, f32 mean, f32 stddev);

	void zero_gradients();
	void optimize(f32 learning_rate);

	std::string name;
	Tensor data;
	Tensor grad;
};

struct Layer
{
	virtual ~Layer() {};
	virtual Tensor& forward(const Tensor& in_act) = 0;

	// back propagates and accumulates gradient parameters
	virtual Tensor& backward(const Tensor& out_grad) = 0;

	virtual void init_params(std::mt19937& gen) {};
	virtual void get_params(std::vector<Parameter*>& params) {};

	virtual void eval() {};
	virtual void train() {};

	Tensor out_act;
	Tensor in_grad;
};

struct LinearLayer : Layer
{
	LinearLayer(u32 in_features, u32 out_features, const std::string& name, bool bias = true);

	Tensor& forward(const Tensor& in_act) override;
	Tensor& backward(const Tensor& out_grad) override;

	void init_params(std::mt19937& gen) override;
	void get_params(std::vector<Parameter*>& params) override;

	u32 in_features, out_features;

	Parameter weights, biases;

	Tensor saved_in_act;
}; 

struct SequentialLayer : Layer
{
	Tensor& forward(const Tensor& in_act) override;
	Tensor& backward(const Tensor& out_grad) override;

	void init_params(std::mt19937& gen) override;
	void get_params(std::vector<Parameter*>& params) override;

	void eval() override;
	void train() override;

protected:
	std::vector<std::unique_ptr<Layer>> layers;
};

struct ReLUActivation : Layer
{
	Tensor& forward(const Tensor& in_act) override;
	Tensor& backward(const Tensor& out_grad) override;
};

struct SoftmaxActivation : Layer
{
	Tensor& forward(const Tensor& in_act) override;
	Tensor& backward(const Tensor& out_grad) override;
};

struct DropoutLayer : Layer
{
	DropoutLayer(std::mt19937& gen, f32 dropout_chance);

	Tensor& forward(const Tensor& in_act) override;
	Tensor& backward(const Tensor& out_grad) override;

	void eval() override;
	void train() override;

	Tensor mask;
	std::mt19937* gen;
	f32 dropout_chance;
	bool eval_mode;
};

struct CrossEntropyLoss
{
	f32 get_loss(const Tensor& act, u32 truth_index);
	Tensor& calculate_loss_gradients(const Tensor& act, u32 truth_index);

	Tensor grad;
};

struct NeuralNetwork : SequentialLayer
{
	template <typename... Layers>
	NeuralNetwork(Layers&&... layers);

	void zero_gradients();
	void optimize(f32 learning_rate);

	void load_state_dict(const StateDict& state_dict);
	void save_state_dict(StateDict& state_dict) const;

	u32 count_params();

private:
	std::vector<Parameter*> params;
};

template<typename ...Layers>
inline NeuralNetwork::NeuralNetwork(Layers && ...layers)
{
	static_assert((std::is_base_of_v<Layer, std::decay_t<Layers>> && ...),
		"All arguments must be derived from Layer");

	(this->layers.emplace_back(std::make_unique<std::decay_t<Layers>>(std::forward<Layers>(layers))), ...);

	this->get_params(params);
}