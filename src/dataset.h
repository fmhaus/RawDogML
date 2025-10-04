#pragma once

#include "types.h"
#include "tensor.h"

#include <memory>
#include <vector>

struct Dataset
{
	Dataset(const char* name);

	usz size() const;
	u32 input_dim() const;
	void print_image(u32 index);

	std::vector<u8> labels;
	std::vector<Tensor> images;
	u32 image_width, image_height;
};