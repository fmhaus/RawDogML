#include "statedict.h"

#include <fstream>

static const u64 MAGIC_NUMBER = 0x21b41241;
static const u64 CURRENT_FILE_VERSION = 1;

template <typename T>
void write_raw(std::ofstream& fw, const T& data)
{
	fw.write(reinterpret_cast<const char*>(&data), sizeof(T));
}

template <typename T>
void read_raw(std::ifstream& fr, T& data)
{
	fr.read(reinterpret_cast<char*>(&data), sizeof(T));
}

StateDict::StateDict()
{
}

StateDict::StateDict(const std::string& file)
{
	load_from_file(file);
}

bool StateDict::contains(const std::string& name) const
{
	return states.find(name) != states.end();
}

void StateDict::put(const std::string& name, Tensor tensor)
{
	states.insert_or_assign(name, tensor);
}

Tensor StateDict::get(const std::string& key) const
{
	return states.at(key);
}

void StateDict::save_to_file(const std::string& file)
{
	std::ofstream fw(file, std::ios::binary);
	if (!fw)
		throw std::runtime_error("Failed to open file!");

	write_raw(fw, MAGIC_NUMBER);
	write_raw(fw, CURRENT_FILE_VERSION);

	write_raw(fw, (u64)states.size());

	for (const auto& pair : states)
	{
		write_raw(fw, (u64)pair.first.length());
		fw.write(pair.first.c_str(), pair.first.length());

		write_raw(fw, (u64)pair.second.size());

		fw.write(reinterpret_cast<const char*>(pair.second.get()), sizeof(f32) * pair.second.size());
	}
}

void StateDict::load_from_file(const std::string& file)
{
	std::ifstream fr(file, std::ios::binary);
	if (!fr)
		throw std::runtime_error("Failed to open file!");

	u64 magic, version;
	read_raw(fr, magic);

	if (magic != MAGIC_NUMBER)
	{
		throw std::runtime_error("Invalid file format!");
	}

	read_raw(fr, version);
	if (version == 1)
	{

		u64 count;
		read_raw(fr, count);

		for (u64 i = 0; i < count; i++)
		{
			u64 str_len, tensor_len;
			read_raw(fr, str_len);

			std::string name(str_len, '\0');
			fr.read(&name[0], str_len);

			read_raw(fr, tensor_len);
			Tensor tensor(tensor_len);
			fr.read(reinterpret_cast<char*>(tensor.get()), tensor_len * sizeof(f32));

			states.try_emplace(std::move(name), std::move(tensor));
		}

	}
	else
	{
		throw std::runtime_error("Unsupported version!");
	}
}
