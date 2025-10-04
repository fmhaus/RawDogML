#include <iostream>

#include <dataset.h>
#include <neuralnetwork.h>

constexpr bool PROGRAM_TRAIN = false; // true for training; false for validating

constexpr usz N_EPOCHS = 50;
constexpr usz ACCUMULATE_STEPS = 10;
constexpr f32 INITIAL_LEARNING_RATE = 0.01;
constexpr f32 LEARNING_RATE_DECAY_FACTOR = 0.95;

const char* STATE_DICT_FILE = RESOURCES_PATH "model_trained.bin";

static u32 get_prediction_index(Tensor& activations)
{
	// get the index with the highest activation
	auto max_it = std::max_element(activations.get(), activations.get() + activations.size());
	return std::distance(activations.get(), max_it);
}

static void validate(NeuralNetwork& network, Dataset& dataset)
{
	u32 correct = 0;
	network.eval();

	for (u32 i = 0; i < dataset.size(); i++)
	{
		auto& result = network.forward(dataset.images[i]);

		if (get_prediction_index(result) == dataset.labels[i])
			correct++;
	}

	f32 percent = (f32)correct / (f32)dataset.size() * 100.0;

	std::cout << "Validated with " << percent << "% Accuracy." << std::endl;
}

static void train(NeuralNetwork& network, Dataset& dataset_train, Dataset& dataset_test, std::mt19937& gen)
{
	CrossEntropyLoss loss_criterion = CrossEntropyLoss();

	f32 learning_rate = INITIAL_LEARNING_RATE;

	// prepare indices list for random list every epoch
	std::vector<u32> train_indices;
	train_indices.resize(dataset_train.size());
	for (u32 i = 0; i < dataset_train.size(); i++)
		train_indices[i] = i;

	for (u32 epoch = 0; epoch < N_EPOCHS; epoch++)
	{
		std::cout << "Learning rate: " << learning_rate << std::endl;

		std::shuffle(train_indices.begin(), train_indices.end(), gen);
		network.zero_gradients();
		network.train();

		f32 total_train_loss = 0.0;
		u32 acc_counter = 0;

		for (u32 i = 0; i < dataset_train.size(); i++)
		{
			u32 train_index = train_indices[i];

			u32 truth_index = dataset_train.labels[train_index];

			auto& result = network.forward(dataset_train.images[train_index]);
			total_train_loss += loss_criterion.get_loss(result, truth_index);

			auto& loss_grad = loss_criterion.calculate_loss_gradients(result, truth_index);
			network.backward(loss_grad);

			acc_counter++;

			if (acc_counter == ACCUMULATE_STEPS || i == dataset_train.size() - 1)
			{
				network.optimize(learning_rate / acc_counter);
				network.zero_gradients();

				acc_counter = 0;
			}
		}

		std::cout << "Epoch " << (epoch + 1) << " training complete. avg_loss: " << (total_train_loss / dataset_train.size()) << std::endl;

		// Validate after every training epoch
		network.eval();

		u32 correct = 0;
		f32 total_test_loss = 0.0;

		for (u32 i = 0; i < dataset_test.size(); i++)
		{
			auto&	 result = network.forward(dataset_test.images[i]);
			total_test_loss += loss_criterion.get_loss(result, dataset_test.labels[i]);

			if (get_prediction_index(result) == dataset_test.labels[i])
				correct++;
		}

		f32 percent = (f32)correct / (f32)dataset_test.size() * 100.0;

		std::cout << "Epoch " << (epoch + 1) << " testing complete. avg_loss: " << (total_test_loss / dataset_train.size()) << " Correct: " << percent << "% " << std::endl;

		learning_rate *= LEARNING_RATE_DECAY_FACTOR;
	}

	// Save final result
	StateDict state;
	network.save_state_dict(state);
	state.save_to_file(STATE_DICT_FILE);
}

int main() 
{
#ifdef _OPENMP
	std::cout << "OpenMP enabled: " << _OPENMP << std::endl;
#endif

	Dataset dataset_test(RESOURCES_PATH "dataset/t10k");
	Dataset dataset_train(RESOURCES_PATH "dataset/train");

	std::cout << "Datasets loaded!" << std::endl; 

	std::random_device rd;
	std::mt19937 gen(rd());

	NeuralNetwork network(
		LinearLayer(dataset_train.input_dim(), 512, "Dense1"),
		ReLUActivation(),
		DropoutLayer(gen, 0.2),

		LinearLayer(512, 256, "Dense2"),
		ReLUActivation(),
		DropoutLayer(gen, 0.2),

		LinearLayer(256, 10, "Dense3"),
		SoftmaxActivation()
	);

	std::cout << "Parameter count: " << network.count_params() << std::endl;

	if (PROGRAM_TRAIN)
	{
		network.init_params(gen);

		train(network, dataset_train, dataset_test, gen);
	}
	else
	{
		StateDict state(STATE_DICT_FILE);
		network.load_state_dict(state);

		validate(network, dataset_test);
	}

	return 0;
}