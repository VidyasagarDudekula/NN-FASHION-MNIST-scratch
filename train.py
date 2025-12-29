import numpy as np
from scipy.special import expit
from data_preprocessor import preprocess_input, preprocess_output, preprocess_line

def relu(x):
    mask = x<0.0
    x[mask] = 0.0
    return x

class NeuralNetwork:
    def __init__(self, inputnodes, hiddennodes1, hiddennodes2, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes1 = hiddennodes1
        self.hnodes2 = hiddennodes2
        self.onodes = outputnodes

        self.lr = learningrate
        self.wih = np.random.normal(0.0, pow(self.hnodes1, -0.5), (self.hnodes1, self.inodes))
        self.whh = np.random.normal(0.0, pow(self.hnodes2, -0.5), (self.hnodes2, self.hnodes1))
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes2))

        self.activation_function_sigmoid = lambda x: expit(x)
        self.activation_function_relu = lambda x: relu(x)

    
    def train(self, input_list, target_list):
        inputs = np.array(input_list, ndmin=2).T
        targets = np.array(target_list, ndmin=2).T

        hidden_inputs1 = np.dot(self.wih, inputs)

        hidden_outputs1 = self.activation_function_relu(hidden_inputs1)

        hidden_inputs2 = np.dot(self.whh, hidden_outputs1)
        hidden_outputs2 = self.activation_function_relu(hidden_inputs2)

        final_inputs = np.dot(self.who, hidden_outputs2)
        final_outputs = self.activation_function_sigmoid(final_inputs)

        output_error = targets - final_outputs
        hidden_error2 = np.dot(self.who.T, output_error)
        hidden_error1 = np.dot(self.whh.T, hidden_error2)

        relu_derivative_2 = (hidden_outputs2 > 0).astype(float)
        relu_derivative_1 = (hidden_outputs1 > 0).astype(float)

        self.who += self.lr * np.dot(output_error * final_outputs * (1-final_outputs), np.transpose(hidden_outputs2))
        self.whh += self.lr * np.dot(hidden_error2 * relu_derivative_2, np.transpose(hidden_outputs1))
        self.wih += self.lr * np.dot(hidden_error1 * relu_derivative_1, np.transpose(inputs))


    def query(self, input_list):
        inputs = np.array(input_list, ndmin=2).T

        hidden_inputs1 = np.dot(self.wih, inputs)

        hidden_outputs1 = self.activation_function_relu(hidden_inputs1)

        hidden_inputs2 = np.dot(self.whh, hidden_outputs1)
        hidden_outputs2 = self.activation_function_relu(hidden_inputs2)

        final_inputs = np.dot(self.who, hidden_outputs2)
        final_outputs = self.activation_function_sigmoid(final_inputs)

        return final_outputs


def test_network(nn_obj):
    if nn_obj is None:
        return None
    test_data_file = open('fashion-mnist_train.csv', 'r')
    test_data_lines = test_data_file.readlines()[1:]
    test_data_file.close()

    test_results = []
    correct = 0
    for index, line in enumerate(test_data_lines):
        data = {}
        data['index'] = index
        line_data = preprocess_line(line)
        input_list = preprocess_input(line_data[0])
        predicted_list = nn_obj.query(input_list)
        index = int(np.argmax(predicted_list).astype(int))
        data['target'] = line_data[1]
        data['predicted'] = index
        data['predicted_list'] = predicted_list.tolist()
        data['result'] = data['target'] == data['predicted']
        if data['result']:
            correct += 1
        test_results.append(data)
    accuracy = correct/len(test_data_lines)
    return accuracy

def train_network(inputnodes, hiddennodes1, hiddennodes2, outputnodes, learningrate, epochs=4):
    nn = NeuralNetwork(inputnodes=inputnodes, hiddennodes1=hiddennodes1, hiddennodes2=hiddennodes2, outputnodes=outputnodes, learningrate=learningrate)
    data_file = open('fashion-mnist_test.csv', 'r')
    data_lines = data_file.readlines()[1:]
    data_file.close()
    stats = {}
    for epoch in range(epochs):
        for line in data_lines:
            line_data = preprocess_line(line)
            input_list = preprocess_input(line_data[0])
            target_list = preprocess_output(line_data[1])
            nn.train(input_list, target_list)
        accuracy = test_network(nn)
        print(f"Epoch:- {epoch}, accuracy:- {accuracy}")
        stats[epoch] = accuracy
    return nn, stats


if __name__ == '__main__':
    import json
    nn_obj, stats = train_network(
        inputnodes=784,
        hiddennodes1=300,
        hiddennodes2=100,
        outputnodes=10,
        learningrate=0.012,
        epochs=10
    )
    with open('basic_stats.json', 'w') as file:
        json.dump(stats, file, indent=4)
    

