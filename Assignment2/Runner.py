import os
import tkinter as tk
from tkinter import filedialog, Label, Button, ttk
import numpy as np
from PIL import Image, ImageTk
import torch
import torch.nn.functional as F
from NN import SimpleNN, train_model, x_test

#Caricamento del modello
model_path = "C:/Users/Edoardo/Documents/GitHub/Digit-recognizer/Assignment2/models/model1.pth"
model = SimpleNN(num_features=784, num_labels=10, hidden_dim=256)

if os.path.exists(model_path):
    print("Modello trovato. Caricamento del modello salvato")
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    print("Nessun modello trovato. Eseguo il training")
    model = train_model(model)


def select_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    if file_path:
        image = Image.open(file_path).convert("L")
        image = image.resize((28, 28))
        img_display = ImageTk.PhotoImage(image)
        label_image.config(image=img_display)
        label_image.image = img_display
        label_image.file_path = file_path


def recognize_digit():
    if hasattr(label_image, 'file_path'):
        # Caricamento immagine e render
        image = Image.open(label_image.file_path).convert("L")
        image = image.resize((28, 28))

        image_array = np.array(image)

        if image_array.mean() > 127:
            image_array = 255 - image_array

        threshold = 50
        image_array = np.where(image_array > threshold, 255, 0)

        image_array = image_array.reshape(1, 784) / 255.0

        mean_target = 0.13
        std_target = 0.3
        current_mean = image_array.mean()
        current_std = image_array.std()

        if current_std != 0:
            image_array = (image_array - current_mean) * (std_target / current_std) + mean_target

        image_array = np.clip(image_array, 0, 1)

        image_tensor = torch.FloatTensor(image_array)
        image_tensor = image_tensor.to(next(model.parameters()).device)

        with torch.no_grad():
            model.eval()
            output = model(image_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)[0].cpu().numpy()
            predicted_label = np.argmax(probabilities)

            result_label.config(text=f"Cifra riconosciuta: {predicted_label}")

            # Aggiorna le barre di probabilità
            for i in range(10):
                prob_bars[i].config(value=probabilities[i] * 100)
                prob_labels[i].config(text=f"{i}: {probabilities[i] * 100:.2f}%")
    else:
        result_label.config(text="Seleziona prima un'immagine!")


# UI
root = tk.Tk()
root.title("Digit Recognizer")
root.geometry("400x600")

label_image = Label(root)
label_image.pack()

btn_select = Button(root, text="Seleziona Immagine", command=select_image)
btn_select.pack()

btn_recognize = Button(root, text="Riconosci Cifra", command=recognize_digit)
btn_recognize.pack()

result_label = Label(root, text="")
result_label.pack()

# Contenitore per le barre di probabilità
frame_probabilities = tk.Frame(root)
frame_probabilities.pack()

prob_bars = []
prob_labels = []
for i in range(10):
    row = tk.Frame(frame_probabilities)
    row.pack(fill='x', padx=10, pady=2)
    label = Label(row, text=f"{i}: 0.00%", width=8, anchor='w')
    label.pack(side='left')
    progress = ttk.Progressbar(row, orient='horizontal', length=200, mode='determinate')
    progress.pack(side='right', fill='x', expand=True)
    prob_labels.append(label)
    prob_bars.append(progress)

root.mainloop()
