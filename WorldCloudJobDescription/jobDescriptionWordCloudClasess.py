
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from wordcloud import WordCloud
import re, os


# Configuración de MAX_LENGTH
MAX_LENGTH = 100
PLOT_INDEX = 30
FILE_NAME_WITHOUT_EXT = ""
DIRECTORY_NAME = ""

STOPWORDS = set([
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because",
    "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during",
    "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's",
    "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our",
    "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that",
    "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've",
    "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "will", "with",
    "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves", "[UNK]"
])

class jobDescriptionOperations:
    """
    Instantiates the jobDescriptionOperations class
    
    Args:
        None.
    
    Returns:
        Procedures
    """   
    
    # Preprocesar el texto (convertir a minúsculas, quitar puntuación, eliminar stopwords)
    def preprocess_text(self, text):
        # Convertir a minúsculas
        text = tf.strings.lower(text)
        
        # Eliminar puntuación
        text = tf.strings.regex_replace(text, r'[^\w\s]', '')
        
        # Dividir el texto en palabras
        words = tf.strings.split(text)
        
        # Usar tf.strings para realizar el filtrado de stopwords
        filtered_words = tf.strings.reduce_join(
            tf.strings.regex_replace(
                words, 
                '|'.join([r'\b' + word + r'\b' for word in STOPWORDS]),  # Eliminar stopwords
                ''
            ),
            separator=' '
        )
        
        return filtered_words

    # Crear el vectorizador de texto
    def create_vectorizer(self, train_sentences):
        vectorizer = tf.keras.layers.TextVectorization(
            standardize=None, 
            output_mode='int', 
            output_sequence_length=MAX_LENGTH,
        )
        vectorizer.adapt(train_sentences)
        return vectorizer

    # Función para contar la frecuencia de las palabras
    def count_word_frequencies(self, dataset, vectorizer):
        word_counts = {}

        for text in dataset:
            # Convertir el texto tokenizado en palabras
            words = self.get_words_from_indices(text.numpy(), vectorizer)

            # Eliminar la palabra [UNK]
            words = [word for word in words if word not in ['[UNK]', '[OOV]']]

            # Contar la frecuencia de cada palabra
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1

        # Crear una tabla con las palabras y sus frecuencias
        word_table = pd.DataFrame(list(word_counts.items()), columns=['Word', 'Frequency'])
        word_table = word_table.sort_values(by='Frequency', ascending=False)

        return word_table

    # Función para revertir índices a palabras
    def get_words_from_indices(self, indices, vectorizer):
        # Obtener el vocabulario del vectorizador
        vocabulary = vectorizer.get_vocabulary(include_special_tokens=True)
        
        # Convertir los índices a palabras
        #words = [vocabulary[index] for index in indices if index < len(vocabulary)]

        words = [vocabulary[index] for index in indices if index < len(vocabulary) and index > 0]
        
        return words

    # Función para crear el gráfico de barras
    def plot_word_frequency(self, word_table, index=PLOT_INDEX):
        # Crear el gráfico de barras para las palabras más frecuentes
        plt.figure(figsize=(10, 6))
        plt.bar(word_table['Word'][:index], word_table['Frequency'][:index], color='skyblue')
        plt.xticks(rotation=90)
        plt.xlabel('Palabras')
        plt.ylabel('Frecuencia')
        plt.title('Palabras más frecuentes')
        plt.show()

    # Función para generar la nube de palabras
    def generate_wordcloud(self, word_table):
        word_freq_dict = dict(zip(word_table['Word'], word_table['Frequency']))

        # Crear la nube de palabras
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq_dict)

        # Mostrar la nube de palabras
        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Nube de palabras')
        plt.show()

    # Función para generar la nube de palabras
    def generate_wordcloud_only(self, word_table):
        word_freq_dict = dict(zip(word_table['Word'], word_table['Frequency']))

        # Crear la nube de palabras
        return WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq_dict)


    # Preprocesar el dataset y crear la tabla de palabras más frecuentes
    def preprocess_and_create_word_table(self, dataset, vectorizer):
        # Aplicar el preprocesamiento
        dataset = dataset.map(lambda x: self.preprocess_text(x))  # Solo texto, sin etiquetas
        
        # Tokenizar el texto
        tokenized_dataset = dataset.map(lambda x: vectorizer(x))

        # Contar las frecuencias de las palabras
        word_table = self.count_word_frequencies(tokenized_dataset, vectorizer)

        return word_table

    # Cargar el archivo de texto (ajusta el path a tu archivo .txt)
    def load_text_file(self, file_path):
        dataset = tf.data.TextLineDataset(file_path)
        return dataset
    
    def splitFileNameWithoutExt(self, tituloventana):

        tituloventana_local = os.path.basename(tituloventana)[:-4]
        fileNameWithoutExt = re.findall(r'[A-Z][^A-Z]*',tituloventana_local)
        fileNameWithoutExt = ' '.join(fileNameWithoutExt)
        #print(f"Titulo ventana: {fileNameWithoutExt}")

        # Set global FILE_NAME_WITHOUT_EXT
        global FILE_NAME_WITHOUT_EXT
        FILE_NAME_WITHOUT_EXT = tituloventana_local

        global DIRECTORY_NAME
        DIRECTORY_NAME = os.path.dirname(tituloventana)

        return fileNameWithoutExt
    
    def save_table_to_csv_png_close_window(self, event, tabla):
        # Guardar los datos de la tabla como archivo CSV
        tabla.to_csv(f"{DIRECTORY_NAME}/{FILE_NAME_WITHOUT_EXT}.csv", index=False)
        print(f"Datos guardados en '{DIRECTORY_NAME}/{FILE_NAME_WITHOUT_EXT}.csv'")

       # Capturar la imagen de la gráfica y guardarla en disco
        plt.savefig(f"{DIRECTORY_NAME}/{FILE_NAME_WITHOUT_EXT}.png")  # Guarda la imagen de la gráfica como 'grafica.png'
        print(f"Gráfica guardada como '{DIRECTORY_NAME}/{FILE_NAME_WITHOUT_EXT}.png'")

        # Cerrar la ventana de la figura
        plt.close()        

    
    def plot_all_in_one(self, dataset, vectorizer, index=PLOT_INDEX, tituloventana=''):
        # Crear una figura y una cuadrícula de subgráficos usando gridspec
        fig = plt.figure(figsize=(12, 8))

        # Ajusta el valor de PLOT_INDEX y FILE_NAME
        global PLOT_INDEX
        PLOT_INDEX = index

        # Set titulo de la grafica
        #fig.suptitle(self.splitFileNameWithoutExt(tituloventana))
       
        # Usar gridspec para controlar el layout
        gs = fig.add_gridspec(2, 2, width_ratios=[1, 2], height_ratios=[5, 1]) 

        # Añadir el gráfico de la tabla (a la izquierda, ocupando toda la altura)

        # Crear la tabla de palabras más frecuentes
        word_table = self.preprocess_and_create_word_table(dataset, vectorizer)
        word_table.reset_index(drop=True, inplace=True)

        #tabla = self.count_word_frequencies(dataset, vectorizer)
        tabla = word_table.head(index)
        ax1 = fig.add_subplot(gs[0, 0])  # Ocupa toda la columna izquierda
        ax1.axis('off')  # Sin ejes en la tabla
        table = ax1.table(cellText=tabla.values, colLabels=tabla.columns, loc='center', cellLoc='center')

        # Crear un botón de descarga
        # Crear el botón debajo de la tabla
        ax_button = fig.add_subplot(gs[1, 0])  # Fila 1, Columna 0
        ax_button.axis('off')  # Desactivar los ejes para el botón

        # Formato de la tabla:
        # Cambiar el fondo del encabezado a gris y el texto a blanco
        for (i, j), cell in table.get_celld().items():
            if i == 0:  # Encabezado de la tabla
                cell.set_text_props(weight='bold', color='white')  # Texto en blanco y negrita
                cell.set_facecolor('gray')  # Fondo gris
            else:
                cell.set_text_props(weight='normal', color='black')  # Texto normal en negro
                cell.set_facecolor('white')  # Fondo blanco para los datos


        ax1.set_title('Tabla de Categorías y Valores', fontsize=14, loc='center', pad=20)

        # Añadir el gráfico de Wordcloud (en la parte superior derecha)
        ax2 = fig.add_subplot(gs[0, 1])  # Fila 0, columna 1
        ax2.imshow(self.generate_wordcloud_only(word_table), interpolation='bilinear')
        ax2.axis('off')  # Sin ejes en la nube de palabras
        ax2.set_title('Wordcloud de Frecuencia de Palabras', fontsize=14, loc='center', pad=20)

        # Crear el gráfico de barras para las palabras más frecuentes
        categorias = word_table['Word'][:index]
        ax3 = fig.add_subplot(gs[1,1])
        ax3.bar(categorias, word_table['Frequency'][:index], color='skyblue', edgecolor='black')
        ax3.set_xticklabels(categorias, rotation=45)
        ax3.set_xticks(categorias)
        ax3.set_title('Palabras más frecuentes', fontsize=14, loc='center', pad=20)
        ax3.set_xlabel('Palabras')
        ax3.set_ylabel('Frecuencia')

        # Crear el botón debajo de la tabla
        ax_button = fig.add_subplot(gs[1, 0])  # Fila 1, Columna 0
        ax_button.axis('off')  # Desactivar los ejes para el botón
        button = Button(ax_button, 'Descargar CSV')
        
        # Personalizar el botón: Cambiar el color de fondo y texto
        button.ax.set_facecolor('lightblue')  # Color de fondo
        button.label.set_color('red')  # Color del texto
        button.label.set_fontsize(14)  # Tamaño de la fuente
        button.label.set_fontweight('bold')  # Fuente en negrita

        # Lanzar el evento on click
        button.on_clicked(lambda event: self.save_table_to_csv_png_close_window(event, tabla))

        # Ajustar el titulo de la ventana
        fig_manager = plt.get_current_fig_manager()
        fig_manager.set_window_title(self.splitFileNameWithoutExt(tituloventana))

        # Ajustar el layout para evitar solapamientos
        plt.tight_layout()
        # Mostrar la gráfica
        plt.show()         
    

class ErrorLeyendoArchivo(Exception):
    def __init__(self, mensaje):
        self.mensaje = mensaje
        super().__init__(self.mensaje)