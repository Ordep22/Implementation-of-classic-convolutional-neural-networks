import tensorflow as tf
from tensorflow.keras.datasets import mnist, cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tensorflow.keras.models import load_model


def load_data(dataset_name):
    if dataset_name == "mnist":
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        input_shape = (28, 28, 1)
    elif dataset_name == "cifar10":
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        input_shape = (32, 32, 3)

    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    return x_train, y_train, x_test, y_test, input_shape


def plot_confusion_matrix(y_true, y_pred):
    confusion_mtx = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_mtx, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Previsão')
    plt.ylabel('Verdadeiro')
    plt.show()


def evaluate_model(model, x_test, y_test):
    _, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Acurácia Geral: {accuracy}")


def build_lenet_model(input_shape, learning_rate=0.001):
    model = Sequential()
    model.add(Conv2D(6, kernel_size=(5, 5), activation='tanh', input_shape=input_shape))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Conv2D(16, kernel_size=(5, 5), activation='tanh'))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(120, activation='tanh'))
    model.add(Flatten())
    model.add(Dense(84, activation='tanh'))
    model.add(Dense(10, activation='softmax'))

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.summary()
    return model


def plot_loss(history):
    plt.plot(history.history['loss'], label='Treinamento')
    plt.plot(history.history['val_loss'], label='Validação')
    plt.xlabel('Épocas')
    plt.ylabel('Perda')
    plt.legend()
    plt.show()


def evaluate_saved_model(model_filename, x_test, y_test):
    # Carregar o modelo treinado a partir do arquivo HDF5
    model = load_model(model_filename)

    # Avaliar o modelo no conjunto de teste
    loss, accuracy = model.evaluate(x_test, y_test)
    return loss, accuracy


if __name__ == "__main__":
    datasets = ["cifar10"]  # ["mnist","cifar10"]
    learning_rates = [0.01, 0.001]
    batch_sizes = [64, 128]
    epochs = 20
    accuracies = []
    model_labels = []

    for dataset_name in datasets:
        x_train, y_train, x_test, y_test, input_shape = load_data(dataset_name)
        for lr in learning_rates:
            for batch_size in batch_sizes:
                print(f"Dataset: {dataset_name}, Taxa de Aprendizado: {lr}, Tamanho do Lote: {batch_size}")
                model = build_lenet_model(input_shape, lr)

                history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
                                    validation_data=(x_test, y_test), verbose=1)

                plot_loss(history)

                y_pred = model.predict(x_test)
                y_pred_classes = np.argmax(y_pred, axis=1)

                y_true_classes = np.argmax(y_test, axis=1)

                evaluate_model(model, x_test, y_test)
                plot_confusion_matrix(y_true_classes, y_pred_classes)

                # Salvar o modelo em disco
                model.save(f"{dataset_name}_lenet_model_lr_{lr}_batch_{batch_size}.h5")

                # Avaliar o modelo diretamente aqui
                loss, accuracy = evaluate_saved_model(f"{dataset_name}_lenet_model_lr_{lr}_batch_{batch_size}.h5",
                                                      x_test, y_test)
                accuracies.append(accuracy)
                model_labels.append(f"{dataset_name} LeNet LR={lr}, Batch={batch_size}")

    # Criar um gráfico de barras para comparar as acurácias dos modelos
    plt.figure(figsize=(10, 6))
    plt.barh(model_labels, accuracies, color='green')
    plt.xlabel('Acurácia')
    plt.title('Acurácia dos Modelos')
    plt.xlim(0.0, 1.0)  # Defina os limites do eixo x
    plt.gca().invert_yaxis()  # Inverta a ordem dos modelos no eixo y
    plt.show()
