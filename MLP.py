import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

np.random.seed(42)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        self.weights1 = np.random.randn(input_size + 1, hidden_size)
        self.weights2 = np.random.randn(hidden_size + 1, output_size)
        self.initial_weights1 = self.weights1.copy()
        self.initial_weights2 = self.weights2.copy()
        self.learning_rate = learning_rate
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

    def forward(self, X):
        self.input_with_bias = np.hstack((X, np.ones((X.shape[0], 1))))
        self.hidden_input = np.dot(self.input_with_bias, self.weights1)
        self.hidden_output = sigmoid(self.hidden_input)
        self.hidden_output_with_bias = np.hstack((self.hidden_output, np.ones((self.hidden_output.shape[0], 1))))
        self.output_input = np.dot(self.hidden_output_with_bias, self.weights2)
        self.output = sigmoid(self.output_input)
        return self.output

    def backward(self, X, y, output):
        self.output_error = y - output #çıkışHata= beklenen-çıktı
        self.output_delta = self.output_error * sigmoid_derivative(self.output_input)  #çıkışHata*(çıkış)*(1-çıkış)
        self.hidden_error = np.dot(self.output_delta, self.weights2[:-1].T)
        self.hidden_delta = self.hidden_error * sigmoid_derivative(self.hidden_input)
        self.weights2 += self.learning_rate * np.dot(self.hidden_output_with_bias.T, self.output_delta)
        self.weights1 += self.learning_rate * np.dot(self.input_with_bias.T, self.hidden_delta)

    def train(self, X, y, epochs):
        errors = []
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)
            error = np.mean(np.abs(y - output))
            errors.append(error)
            if epoch % 1000 == 0:
                print(f"Epoch {epoch}: Hata = {error:.6f}")
        total_error = np.sum(errors)
        print(f"\nEğitim Sonrası Toplam Hata: {total_error:.6f}")
        return errors

    def visualize_network(self, weights=True, is_initial=True):
        G = nx.DiGraph()
        input_nodes = [f"X{i + 1}" for i in range(self.input_size)] + ["Bias1"]
        hidden_nodes = [f"H{i + 1}" for i in range(self.hidden_size)] + ["Bias2"]
        output_nodes = [f"Y{i + 1}" for i in range(self.output_size)]
        pos = {}

        for i, node in enumerate(input_nodes):
            G.add_node(node)
            pos[node] = (0, 2 * ((len(input_nodes) - 1) / 2 - i))

        for i, node in enumerate(hidden_nodes):
            G.add_node(node)
            pos[node] = (1, 2 * ((len(hidden_nodes) - 1) / 2 - i))

        for i, node in enumerate(output_nodes):
            G.add_node(node)
            pos[node] = (2, 2 * ((len(output_nodes) - 1) / 2 - i))

        edge_labels = {}

        for i in range(len(input_nodes)):
            for j in range(len(hidden_nodes) - 1):
                G.add_edge(input_nodes[i], hidden_nodes[j])
                if weights:
                    weight = self.initial_weights1[i, j] if is_initial else self.weights1[i, j]
                    edge_labels[(input_nodes[i], hidden_nodes[j])] = f"{weight:.2f}"

        for i in range(len(hidden_nodes)):
            for j in range(len(output_nodes)):
                G.add_edge(hidden_nodes[i], output_nodes[j])
                if weights:
                    weight = self.initial_weights2[i, j] if is_initial else self.weights2[i, j]
                    edge_labels[(hidden_nodes[i], output_nodes[j])] = f"{weight:.2f}"

        plt.figure(figsize=(15, 12))
        status = "BAŞLANGIÇ" if is_initial else "EĞİTİM SONRASI"
        plt.suptitle(f'YAPAY SİNİR AĞI YAPISI\n({status} AĞIRLIKLAR VE BİASLAR)', fontsize=16, y=0.98)

        nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, arrowsize=20, font_size=10, font_weight='bold', arrows=True, edge_color='gray', width=1.0)

        bias_edge_labels = {}
        for j in range(len(hidden_nodes) - 1):
            bias_weight = self.initial_weights1[-1, j] if is_initial else self.weights1[-1, j]
            bias_edge_labels[("Bias1", hidden_nodes[j])] = f"Bias: {bias_weight:.2f}"

        for j in range(len(output_nodes)):
            bias_weight = self.initial_weights2[-1, j] if is_initial else self.weights2[-1, j]
            bias_edge_labels[("Bias2", output_nodes[j])] = f"Bias: {bias_weight:.2f}"

        if weights:
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, label_pos=0.3, bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

        nx.draw_networkx_edge_labels(G, pos, edge_labels=bias_edge_labels, font_size=10, label_pos=0.6, bbox=dict(facecolor='lightyellow', edgecolor='none', alpha=0.8))

        plt.axis('off')
        plt.show()

X = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
y = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

while True:
    try:
        hidden_neurons = int(input("Gizli katmandaki nöron sayısını giriniz: "))
        if hidden_neurons > 0:
            break
        else:
            print("Lütfen pozitif bir sayı giriniz.")
    except ValueError:
        print("Lütfen geçerli bir sayı giriniz.")

nn = NeuralNetwork(input_size=4, hidden_size=hidden_neurons, output_size=2, learning_rate=0.1)

print("\nBAŞLANGIÇ AĞIRLIKLARI:")
print("\nGiriş-Gizli Katman Ağırlıkları (weights1):")
print(nn.initial_weights1)
print("\nGizli-Çıkış Katman Ağırlıkları (weights2):")
print(nn.initial_weights2)

nn.visualize_network(weights=True, is_initial=True)

epochs = 10000
errors = nn.train(X, y, epochs)

print("\nEğitim Sonrası Ağırlıklar:")
print("\nGiriş-Gizli Katman Ağırlıkları (weights1):")
print(nn.weights1)
print("\nGizli-Çıkış Katman Ağırlıkları (weights2):")
print(nn.weights2)

nn.visualize_network(weights=True, is_initial=False)

plt.plot(errors)
plt.title('Eğitim Hatası')
plt.xlabel('Epoch')
plt.ylabel('Hata')
plt.show()
