from jobDescriptionWordCloudClasess import jobDescriptionOperations, ErrorLeyendoArchivo

import tkinter as tk
from tkinter import filedialog

def main():

    try:
        jdop = jobDescriptionOperations()

        # Crear la ventana principal de Tkinter (no se mostrará)
        root = tk.Tk()
        root.withdraw()  # Oculta la ventana principal

        # Mostrar el cuadro de diálogo para seleccionar un archivo
        file_path = filedialog.askopenfilename(title="Select a file with the Job Desciption to analyze", filetypes=[("Text Files", "*.txt")])

        # Mostrar la ruta del archivo seleccionado
        if file_path:
            jobDescriptionFile = file_path
            print("Archivo seleccionado:", file_path)
        else:
            raise ErrorLeyendoArchivo("No se seleccionó ningún archivo.")

        dataset = jdop.load_text_file(jobDescriptionFile)

        # Crear y adaptar el vectorizador
        vectorizer = jdop.create_vectorizer(dataset)

        jdop.plot_all_in_one(dataset, vectorizer, 20, jobDescriptionFile)
    
    except ErrorLeyendoArchivo as e:
        print(f"Se ha producido un error: {e}")

if __name__ == '__main__':
    main()