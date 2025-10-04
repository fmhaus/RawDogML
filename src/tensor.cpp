#include "tensor.h"

#include <cassert>

TensorBuffer::TensorBuffer(usz size)
	: memory(), size(size)
{
	if (size)
		memory = std::make_unique<f32[]>(size);
}

Tensor::Tensor()
	: buffer()
{
}

Tensor::Tensor(usz size)
	: buffer(size > 0 ? std::make_shared<TensorBuffer>(size) : nullptr)
{
}

usz Tensor::size() const
{
	if (buffer)
		return buffer->size;
	return 0;
}

f32& Tensor::operator[](usz index)
{
	return buffer->memory[index];
}

const f32& Tensor::operator[](usz index) const
{
	return buffer->memory[index];
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

Tensor::operator bool() const
{
	return size() > 0;
}

void Tensor::set_zeros()
{
	if (buffer)
		memset(get(), 0, sizeof(f32) * size());
}

void Tensor::fill(f32 value)
{
	#pragma omp parallel for
	for (s32 i = 0; i < size(); i++)
	{
		get()[i] = value;
	}
}

Tensor Tensor::steal_memory()
{
	Tensor t = *this;
	this->buffer.reset();
	return std::move(t);
}
