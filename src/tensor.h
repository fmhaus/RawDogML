#pragma once

#include "types.h"

#include <memory>

struct TensorBuffer
{
	TensorBuffer(usz size);

	std::unique_ptr<f32[]> memory;
	usz size;
};

struct Tensor
{
	Tensor();
	Tensor(usz size);

	usz size() const;

	f32& operator[](usz index);
	const f32& operator[](usz index) const;

	f32* get();
	const f32* get() const;

	explicit operator bool() const;

	void set_zeros();
	void fill(f32 value);

private:
	std::shared_ptr<TensorBuffer> buffer;
};