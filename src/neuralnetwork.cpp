#include "neuralnetwork.h"

#include <cassert>
#include <cmath>
#include <iostream>

inline static void ensure_tensor_size(Tensor* tensor, s32 size)
{
	if (tensor->size() < size)
		*tensor = Tensor(size);
}

Parameter::Parameter(u32 size, const std::string& name)
	: name(name),
	data(size),
	grad()
{ }

void Parameter::init_normal_dist(std::mt19937& gen, f32 mean, f32 stddev)
{
	std::normal_distribution<float> dist(mean, stddev);
	for (s32 i = 0; i < data.size(); i++)
	{
		data[i] = dist(gen);
	}
}

void Parameter::zero_gradients()
{
	ensure_tensor_size(&grad, data.size());
	grad.set_zeros();
}

void Parameter::optimize(f32 learning_rate)
{
	#pragma omp parallel for
	for (s32 i = 0; i < data.size(); i++)
	{
		data[i] -= grad[i] * learning_rate;
	}
}

LinearLayer::LinearLayer(u32 in_features, u32 out_features, const std::string& name, bool bias)
	: in_features(in_features), 
	out_features(out_features), 
	weights(in_features * out_features, name + ".weights"),
	biases(bias ? out_features : 0, name + ".biases")
{
}

Tensor& LinearLayer::forward(const Tensor& in_act)
{
	assert(in_act.size() == in_features);
	saved_in_act = in_act;

	ensure_tensor_size(&out_act, out_features);

	// zero out_act
	out_act.set_zeros();

	bool use_biases = biases.data.size() > 0;

	// do linear feed forward
	#pragma omp parallel for
	for (s32 j = 0; j < out_features; j++)
	{
		f32 sum = use_biases ? biases.data[j] : 0.0;

		for (s32 i = 0; i < in_features; i++)
		{
			sum += weights.data[j * in_features + i] * in_act[i];
		}
		out_act[j] = sum;
	}

	return out_act;
}

Tensor& LinearLayer::backward(const Tensor& out_grad)
{
	assert(out_grad.size() >= out_features);
	assert(weights.grad.size() == weights.data.size());
	assert(biases.grad.size() == biases.data.size());

	ensure_tensor_size(&in_grad, in_features);

	in_grad.set_zeros();

	if (biases.data.size() > 0)
	{
		for (s32 j = 0; j < out_features; j++)
			biases.grad[j] += out_grad[j];
	}

	#pragma omp parallel for
	for (s32 j = 0; j < out_features; j++)
	{
		for (s32 i = 0; i < in_features; i++)
		{
			weights.grad[j * in_features + i] += saved_in_act[i] * out_grad[j];

			in_grad[i] += weights.data[j * in_features + i] * out_grad[j];
		}
	}

	return in_grad;
}

void LinearLayer::init_params(std::mt19937& gen)
{
	weights.init_normal_dist(gen, 0.0, std::sqrtf(2.0 / (f32)in_features));
	biases.data.set_zeros();
}

void LinearLayer::get_params(std::vector<Parameter*>& params)
{
	params.push_back(&weights);
	if (biases.data.size())
		params.push_back(&biases);
}

Tensor& SequentialLayer::forward(const Tensor& in_act)
{
	if (layers.size() == 0)
	{ 
		out_act = in_act;
	}
	else if (layers.size() == 1)
	{
		layers.front()->forward(in_act);
		out_act = layers.front()->out_act;
	}
	else
	{
		const Tensor* in_ptr = &in_act;
		for (auto& layer : layers)
		{
			in_ptr = &layer->forward(*in_ptr);
		}
		out_act = *in_ptr;
	}

	return out_act;
}

Tensor& SequentialLayer::backward(const Tensor& out_grad)
{
	if (layers.size() == 0)
	{
		in_grad = out_grad;
	}
	else if (layers.size() == 1)
	{
		layers.front()->backward(out_grad);
		in_grad = layers.front()->in_grad;
	}
	else
	{
		const Tensor* out_ptr = &out_grad;
		for (s32 i = layers.size() - 1; i >= 0; i--)
		{
			out_ptr = &layers[i]->backward(*out_ptr);
		}
		in_grad = *out_ptr;
	}

	return in_grad;
}

void SequentialLayer::init_params(std::mt19937& gen)
{
	for (auto& layer : layers)
		layer->init_params(gen);
}

void SequentialLayer::get_params(std::vector<Parameter*>& params)
{
	for (auto& layer : layers)
		layer->get_params(params);
}

void SequentialLayer::eval()
{
	for (auto& layer : layers)
		layer->eval();
}

void SequentialLayer::train()
{
	for (auto& layer : layers)
		layer->train();
}

Tensor& ReLUActivation::forward(const Tensor& in_act)
{
	ensure_tensor_size(&out_act, in_act.size());

	#pragma omp parallel for
	for (s32 i = 0; i < in_act.size(); i++)
	{
		out_act[i] = in_act[i] > 0 ? in_act[i] : 0;
	}

	return out_act;
}

Tensor& ReLUActivation::backward(const Tensor& out_grad)
{
	ensure_tensor_size(&in_grad, out_grad.size());

	#pragma omp parallel for
	for (s32 i = 0; i < out_grad.size(); i++)
	{
		in_grad[i] = (out_act[i] > 0 ? 1 : 0) * out_grad[i];
	}

	return in_grad;
}

Tensor& SoftmaxActivation::forward(const Tensor& in_act)
{
	if (in_act.size() == 0)
		return out_act;

	ensure_tensor_size(&out_act, in_act.size());

	f32 max_val = in_act[0];
	for (s32 i = 1; i < in_act.size(); i++)
	{
		max_val = std::max(max_val, in_act[i]);
	}

	f32 sum = 0.0;
	#pragma omp parallel for reduction(+:sum)
	for (s32 i = 0; i < in_act.size(); i++)
	{
		out_act[i] = std::expf(in_act[i] - max_val);
		sum += out_act[i];
	}

	#pragma omp parallel for
	for (s32 i = 0; i < in_act.size(); i++)
	{
		out_act[i] /= sum;
	}

	return out_act;
}

Tensor& SoftmaxActivation::backward(const Tensor& out_grad)
{
	ensure_tensor_size(&in_grad, out_grad.size());

	in_grad.set_zeros();

	#pragma omp parallel for
	for (s32 j = 0; j < out_grad.size(); j++)
	{
		f32 sum = 0.0;
		for (s32 i = 0; i < out_grad.size(); i++)
		{
			f32 softmax_gradient = (i == j)
				? out_act[i] * (1.0 - out_act[i])
				: -out_act[i] * out_act[j];

			sum += out_grad[i] * softmax_gradient;
		}

		in_grad[j] = sum;
	}

	return in_grad;
}

DropoutLayer::DropoutLayer(std::mt19937* gen, f32 dropout_chance)
	: mask(), gen(gen), dropout_chance(dropout_chance), eval_mode(true)
{
}

Tensor& DropoutLayer::forward(const Tensor& in_act)
{
	if (eval_mode)
	{
		out_act = in_act;
		return out_act;
	}

	ensure_tensor_size(&out_act, in_act.size());
	ensure_tensor_size(&mask, in_act.size());

	std::uniform_real_distribution<f32> dist(0.0, 1.0);
	f32 pass_chance = 1.0 - dropout_chance;
	f32 scaling = 1.0 / pass_chance;
	
	// cant parallelize because of random values.
	// big layers should instead use thread local random generation
	for (s32 i = 0; i < in_act.size(); i++)
	{
		mask[i] = dist(*gen) < pass_chance ? scaling : 0.0;
	}

	#pragma omp parallel for
	for (s32 i = 0; i < in_act.size(); i++)
	{
		out_act[i] = mask[i] * in_act[i];
	}

	return out_act;
}

Tensor& DropoutLayer::backward(const Tensor& out_grad)
{
	if (eval_mode)
	{
		in_grad = out_grad;
	}
	else
	{
		ensure_tensor_size(&in_grad, out_grad.size());

		#pragma omp parallel for
		for (s32 i = 0; i < out_grad.size(); i++)
		{
			in_grad[i] = mask[i] * out_grad[i];
		}
	}
	return in_grad;
}

void DropoutLayer::eval()
{
	eval_mode = true;
	mask = Tensor();
}

void DropoutLayer::train()
{
	eval_mode = false;
	// in eval mode, the layer reuses and passed the activations (and grad) through.
	// when switching to training, reset in_grad and out_act to ensure new memory is allocated to it.
	in_grad = Tensor();
	out_act = Tensor();
}

f32 CrossEntropyLoss::get_loss(const Tensor& act, u32 truth_index)
{
	return -std::logf(act[truth_index]);
}

Tensor& CrossEntropyLoss::calculate_loss_gradients(const Tensor& act, u32 truth_index)
{
	ensure_tensor_size(&grad, act.size());

	grad.set_zeros();
	grad[truth_index] = -1.0 / act[truth_index];

	return grad;
}

void NeuralNetwork::zero_gradients()
{
	for (auto& param : params)
	{
		ensure_tensor_size(&param->grad, param->data.size());
		param->grad.set_zeros();
	}
}

void NeuralNetwork::optimize(f32 learning_rate)
{
	for (auto& param : params)
	{
		param->optimize(learning_rate);
	}
}

void NeuralNetwork::load_state_dict(const StateDict& state_dict)
{
	for (auto& param : params)
	{
		if (state_dict.contains(param->name))
		{
			const Tensor& saved_tensor = state_dict.get(param->name);
			if (saved_tensor.size() != param->data.size())
			{
				std::cerr << "Tensor dimension mismatch. Key: " << param->name << " Saved: " << saved_tensor.size() << " Required: " << param->data.size() << std::endl;
			}
			else
			{
				param->data = saved_tensor;
			}
		}
		else
		{
			std::cerr << "Missing key in state dict: " << param->name << std::endl;
		}
	}
}

void NeuralNetwork::save_state_dict(StateDict& state_dict) const
{
	for (auto& param : params)
	{
		state_dict.put(param->name, param->data);
	}
}

u32 NeuralNetwork::count_params()
{
	u32 count = 0;
	for (auto& param : params)
		count += param->data.size();
	return count;
}