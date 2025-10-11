#include "tensor.h"

#include <cassert>
#include <sstream>

TensorShape::TensorShape()
	: dims()
{
}

TensorShape::TensorShape(std::vector<u64>&& dims)
	: dims(dims)
{
}

usz TensorShape::numel() const
{
	usz p = 1;
	for (usz dim : dims)
		p *= dim;
	return p;
}

std::string TensorShape::to_string() const
{
	std::ostringstream oss;
	oss << "[";
	for (size_t i = 0; i < dims.size(); ++i) {
		oss << dims[i];
		if (i < dims.size() - 1) oss << ", ";
	}
	oss << "]";
	return oss.str();
}

bool TensorShape::operator==(const TensorShape& other) const
{
	return dims == other.dims;
}

bool TensorShape::operator!=(const TensorShape& other) const
{
	return dims != other.dims;
}

TensorBuffer::TensorBuffer(usz capacity, TensorShape&& shape)
	: memory(), capacity(capacity), shape(shape)
{
	if (capacity)
		memory = std::make_unique<f32[]>(capacity);
}


void Tensor::copy(const Tensor& source, Tensor& dest)
{
	assert(source.numel() == dest.numel());
	#pragma omp parallel for
	for (s64 i = 0; i < source.numel(); i++)
		dest[i] = source[i];
}

Tensor::Tensor()
	: buffer()
{
}

Tensor::Tensor(usz size)
	: buffer(size > 0 ? std::make_shared<TensorBuffer>(size, std::move(TensorShape(size))) : nullptr)
{
}

Tensor::Tensor(TensorShape&& shape)
	: buffer(std::make_shared<TensorBuffer>(shape.numel(), std::move(shape)))
{
}

Tensor::Tensor(const TensorShape& shape)
	: buffer(std::make_shared<TensorBuffer>(shape.numel(), std::move(TensorShape(shape))))
{
}

usz Tensor::size() const
{
	if (buffer)
		return buffer->capacity;
	return 0;
}

const TensorShape& Tensor::shape() const
{
	if (buffer)
	{
		return buffer->shape;
	}
	else
	{
		static const TensorShape empty_shape;
		return empty_shape;
	}
}

f32* Tensor::get()
{
	if (buffer)
		return buffer->memory.get();
	return nullptr;
}

const f32* Tensor::get() const
{
	if (buffer)
		return buffer->memory.get();
	return nullptr;
}

void Tensor::set_zeros()
{
	fill(0);
}

void Tensor::fill(f32 value)
{
	#pragma omp parallel for
	for (s32 i = 0; i < size(); i++)
	{
		get()[i] = value;
	}
}

Tensor Tensor::clone()
{
	if (buffer)
	{
		Tensor tensor(shape());
		Tensor::copy(*this, tensor);
		return tensor;
	}
	else
	{
		return Tensor();
	}
}
