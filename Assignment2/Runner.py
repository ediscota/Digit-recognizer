import os
import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import torch
import torchvision.transforms as transforms
from NN import SimpleNN  # Assicurati di avere il modello definito in model.py

# Caricamento del modello
model_path = "C:/Users/Edoardo/Desktop/Università/AI/Assignment2/models/best_model.pth"

# Creiamo il modello
model = SimpleNN(num_features=784, num_labels=10, hidden_dim=5)  # Assicurati che hidden_dim sia corretto

# Se il modello esiste, lo carichiamo, altrimenti facciamo il training
if os.path.exists(model_path):
    print("Modello trovato! Caricamento del modello salvato...")
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Modalità di valutazione, nessun training necessario
else:
    print("Nessun modello trovato! Eseguo il training...")
    #train_model(model)  # Supponiamo che questa sia la tua funzione di training
    torch.save(model.state_dict(), model_path)
    print("Training completato e modello salvato!")


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
        image = Image.open(label_image.file_path).convert("L")
        image = image.resize((28, 28))
        transform = transforms.ToTensor()
        image = transform(image).view(-1, 784)  # Flatten a 1x784

        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)
            result_label.config(text=f"Cifra riconosciuta: {predicted.item()}")
    else:
        result_label.config(text="Seleziona prima un'immagine!")


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

result_label = Label(root, text="")
result_label.pack()

root.mainloop()
