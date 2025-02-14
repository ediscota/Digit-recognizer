import os
import tkinter as tk
from tkinter import filedialog, Label, Button
import numpy as np
from PIL import Image, ImageTk
import torch
import torch.nn.functional as F
from NN import SimpleNN, train_model, x_test

# Caricamento del modello
model_path = "C:/Users/Edoardo/Documents/GitHub/Digit-recognizer/Assignment2/models/model1.pth"
model = SimpleNN(num_features=784, num_labels=10, hidden_dim=256)

if os.path.exists(model_path):
    print("Modello trovato! Caricamento del modello salvato...")
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict']) #?
    print(f"Modello caricato dall'epoch {checkpoint['epoch']} con accuracy {checkpoint['val_acc']:.4f}")

    # Test immediato del modello, da eliminare
    model.eval()
    test_input = torch.randn(1, 784)  # Input casuale per test
    with torch.no_grad():
        output = model(test_input)
        probs = F.softmax(output, dim=1)
        print("\nTest del modello caricato:")
        print(f"Probabilità output: {probs[0].numpy()}")
else:
    print("Nessun modello trovato! Eseguo il training...")
    model = train_model()


def select_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    if file_path:
        image = Image.open(file_path).convert("L")  # Converti in scala di grigi
        image = image.resize((28, 28))  # Ridimensiona a 28x28
        img_display = ImageTk.PhotoImage(image)
        label_image.config(image=img_display)
        label_image.image = img_display
        label_image.file_path = file_path


def recognize_digit():
    if hasattr(label_image, 'file_path'):
        # 1. Carica e prepara l'immagine
        image = Image.open(label_image.file_path).convert("L")
        image = image.resize((28, 28))

        # 2. Converti in array numpy per il preprocessing
        image_array = np.array(image)

        # 3. Inverti l'immagine se necessario
        if image_array.mean() > 127:
            image_array = 255 - image_array

        # 4. Applica soglia per aumentare il contrasto
        threshold = 50
        image_array = np.where(image_array > threshold, 255, 0)

        # Visualizza l'immagine preprocessata prima della normalizzazione, da eliminare
        preprocessed_image = Image.fromarray(image_array.reshape(28, 28).astype('uint8'))
        preprocessed_display = ImageTk.PhotoImage(
            preprocessed_image.resize((100, 100)))  # Ingrandisci per migliore visualizzazione

        # Aggiorna il label per mostrare l'immagine preprocessata, da eliminare
        preprocessed_label.config(image=preprocessed_display)
        preprocessed_label.image = preprocessed_display

        # 5. Normalizza e prepara per il modello
        image_array = image_array.reshape(1, 784) / 255.0

        # 6. Applica correzione della distribuzione
        mean_target = 0.13
        std_target = 0.3
        current_mean = image_array.mean()
        current_std = image_array.std()

        if current_std != 0:
            image_array = (image_array - current_mean) * (std_target / current_std) + mean_target

        image_array = np.clip(image_array, 0, 1)

        print(f"Statistiche finali dell'input:")
        print(f"Min: {image_array.min()}, Max: {image_array.max()}")
        print(f"Media: {image_array.mean()}, Std: {image_array.std()}")

        image_tensor = torch.FloatTensor(image_array)
        image_tensor = image_tensor.to(next(model.parameters()).device)

        with torch.no_grad(): #eliminabile
            model.eval()
            output = model(image_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            predicted_label = torch.argmax(probabilities, dim=1).item()

            prob_text = "\n".join([f"{i}: {probabilities[0][i].item() * 100:.2f}%" for i in range(10)])
            result_label.config(text=f"Cifra riconosciuta: {predicted_label}\n\nProbabilità:\n{prob_text}")

            print(f"\nProbabilità per ogni cifra:")
            print(probabilities[0].cpu().numpy())
    else:
        result_label.config(text="Seleziona prima un'immagine!")


def test_with_mnist_image():#da eliminare
    # Prendi una singola immagine dal test set
    test_image = x_test[0:1]  # Prende la prima immagine

    # Converti in tensor
    image_tensor = torch.FloatTensor(test_image)
    image_tensor = image_tensor.to(next(model.parameters()).device)

    print("Test con immagine MNIST originale:")
    print(f"Min: {test_image.min()}, Max: {test_image.max()}")
    print(f"Media: {test_image.mean()}, Std: {test_image.std()}")

    with torch.no_grad():
        model.eval()
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        predicted_label = torch.argmax(probabilities, dim=1).item()

        print(f"\nProbabilità per ogni cifra (immagine MNIST):")
        print(probabilities[0].cpu().numpy())

# Creazione dell'interfaccia
root = tk.Tk()
root.title("Digit Recognizer")
root.geometry("400x500")

label_image = Label(root)
label_image.pack()

btn_select = Button(root, text="Seleziona Immagine", command=select_image)
btn_select.pack()

btn_recognize = Button(root, text="Riconosci Cifra", command=recognize_digit)
btn_recognize.pack()

btn_test_mnist = Button(root, text="Test con MNIST", command=test_with_mnist_image)
btn_test_mnist.pack()

result_label = Label(root, text="")
result_label.pack()

preprocessed_label = Label(root)
preprocessed_label.pack()

root.mainloop()
