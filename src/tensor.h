#pragma once

#include "types.h"

#include <memory>
#include <vector>
#include <cassert>
#include <string>

struct TensorShape
{
	TensorShape();
	TensorShape(std::vector<u64>&& dims); 

	template<typename... Dims>
	TensorShape(usz first, Dims... list)
		: dims{ first, list... }
	{
	}

	inline usz& operator[](usz index) { return dims[index];	}
	const usz& operator[](usz index) const { return dims[index]; };
	usz length() const { return dims.size(); };

	usz numel() const;

	std::string to_string() const;

	bool operator==(const TensorShape& other) const;
	bool operator!=(const TensorShape& other) const;

	std::vector<usz> dims;
};

struct TensorBuffer
{
	TensorBuffer(usz capacity, TensorShape&& shape);

	std::unique_ptr<f32[]> memory;
	TensorShape shape;
	usz capacity;
};

struct Tensor
{
	static void copy(const Tensor& source, Tensor& dest);

	Tensor();
	Tensor(usz size);
	Tensor(TensorShape&& shape);
	Tensor(const TensorShape& shape);

	template<typename... Dims>
	Tensor(usz first, Dims... dims)
		: Tensor(std::move(TensorShape({ first, dims... })))
	{}

	usz size() const;
	inline usz numel() const { return size(); };

	const TensorShape& shape() const;

	template <typename... Indices>
	inline f32& operator[](Indices... indices)
	{
		return buffer->memory[element_index(std::forward<Indices>(indices)...)];
	}

	template <typename... Indices>
	inline const f32& operator[](Indices... indices) const
	{
		return buffer->memory[element_index(std::forward<Indices>(indices)...)];
	}

	f32* get();
	const f32* get() const;

	explicit inline operator bool() const { return (bool)buffer; };

	void set_zeros();
	void fill(f32 value);
	Tensor clone();

private:
	std::shared_ptr<TensorBuffer> buffer;

	template<typename... Indices>
	inline usz element_index(usz first, Indices... indices) const
	{
		usz dim_index = 1;
		usz result = first;

		((result = result * buffer->shape[dim_index++] + indices), ...);
		assert(dim_index == buffer->shape.length() || dim_index == 1);
		return result;
	}
};