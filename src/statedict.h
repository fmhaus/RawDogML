#pragma once

#include "tensor.h"

#include <unordered_map>
#include <string>

struct StateDict
{
	StateDict();
	StateDict(const std::string& file);

	bool contains(const std::string& name) const;
	void put(const std::string& name, Tensor tensor);
	Tensor get(const std::string& key) const;

	void save_to_file(const std::string& file);
	void load_from_file(const std::string& file);

private:
	std::unordered_map<std::string, Tensor> states;
};