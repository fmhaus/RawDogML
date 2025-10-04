#include <dataset.h>

#include <string>
#include <memory>
#include <fstream>
#include <iostream>
#include <bit>
#include <stdexcept>

static u32 read_big_endian(std::ifstream& ifs)
{
	u32 value;
	ifs.read(reinterpret_cast<char*>(&value), sizeof(u32));

	// bytes in dataset are in big endian, swap if device is little endian
	if (std::endian::native == std::endian::little)
		value = std::byteswap(value);

	return value;
}

Dataset::Dataset(const char* name)
{
	std::string images_file = std::string(name) + "-images.idx3-ubyte";
	std::string labels_file = std::string(name) + "-labels.idx1-ubyte";

	std::ifstream f_images(images_file, std::ios::binary);
	if (!f_images.is_open())
	{
		throw std::runtime_error(std::string("Failed to open images file (") + name + ")");
	}

	u32 magic_images = read_big_endian(f_images);
	u32 num_images = read_big_endian(f_images);
	this->image_height = read_big_endian(f_images);
	this->image_width = read_big_endian(f_images);

	if (magic_images != 2051)
		throw std::runtime_error("Invalid images magic number.");

	std::ifstream f_labels(labels_file, std::ios::binary);
	if (!f_labels.is_open())
	{
		throw std::runtime_error(std::string("Failed to open labels file (") + name + ")");
	}

	u32 magic_labels = read_big_endian(f_labels);
	u32 num_labels = read_big_endian(f_labels);

	if (magic_labels != 2049)
		throw std::runtime_error("Invalid labels magic number");

	if (num_images != num_labels)
		throw std::runtime_error("Images and Labels count are not matching");

	images.reserve(num_images);
	labels.reserve(num_labels);

	// temporary buffer for reading from file
	u32 image_dim = this->image_height * this->image_width;
	std::unique_ptr<u8[]> read_buffer = std::make_unique<u8[]>(image_dim);

	for (u32 i = 0; i < num_images; i++)
	{
		// read into buffer
		f_images.read(reinterpret_cast<char*>(read_buffer.get()), image_dim);

		// convert directly to float and scale
		Tensor tensor(image_dim);
		for (u32 j = 0; j < image_dim; j++)
			tensor.get()[j] = (f32) read_buffer[j] / 255.0f;

		images.push_back(std::move(tensor));
	}

	for (u32 i = 0; i < num_labels; i++)
	{
		u8 label;
		f_labels.read(reinterpret_cast<char*>(&label), sizeof(label));
		labels.push_back(label);
	}

}

usz Dataset::size() const
{
	return labels.size();
}

u32 Dataset::input_dim() const
{
	return image_height * image_width;
}

void Dataset::print_image(u32 index)
{
	// print image in console
	Tensor& tensor = images[index];
	for (u32 y = 0; y < image_height; y++)
	{
		for (u32 x = 0; x < image_width; x++)
		{
			if (tensor[y * image_width + x] > 0.5f)
				std::cout << "# ";
			else
				std::cout << ". ";
		}
		std::cout << std::endl;
	}
}