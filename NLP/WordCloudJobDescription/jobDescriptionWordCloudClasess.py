
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from wordcloud import WordCloud
import re, os
import nltk
from nltk.corpus import stopwords
from jobDescriptionExceptions import ErrorGettingStopWords



# MAX_LENGTH Configuration

MAX_LENGTH = 100
PLOT_INDEX = 30
FILE_NAME_WITHOUT_EXT = ""
DIRECTORY_NAME = ""

class jobDescriptionOperations:
    """
    Instantiates the jobDescriptionOperations class
    
    Args:
        language: Language used to get the stopwords.
    
    Returns:
        Procedures
    """   
    # Class initializacion
    def __init__(self, language='en'):
        self.language = language
        self.stopwords = None

        print("Language: ", self.language)

        self.get_stopwords()

    
    # Setup the language to get the stop words used
    def get_stopwords(self):

        try:
            nltk.download('stopwords')

            if(self.language.lower() =='en'):
                self.stopwords = stopwords.words('english')
            elif(self.language.lower() =='es'):
                self.stopwords = stopwords.words('spanish')
            else:
                raise ErrorGettingStopWords("There aren't language defined. By default, the system will use english as default language")
        
        except ErrorGettingStopWords as e:
            print(f"An error has occurred: {e}")
            self.stopwords = stopwords.words('english')

    
    # Preprocess the text (convert to lowercase, remove punctuation, remove stopwords)
    def preprocess_text(self, text):
        # Convert to lowercase
        text = tf.strings.lower(text)
        
        # Delete punctuation
        text = tf.strings.regex_replace(text, r'[^\w\s]', '')
        
        # Split text into words
        words = tf.strings.split(text)
        
        # Using tf.strings to perform stopword filtering
        filtered_words = tf.strings.reduce_join(
            tf.strings.regex_replace(
                words, 
                '|'.join([r'\b' + word + r'\b' for word in self.stopwords]),  # Delete stopwords
                ''
            ),
            separator=' '
        )
        
        return filtered_words

    # Create the text vectorizer.
    def create_vectorizer(self, train_sentences):
        vectorizer = tf.keras.layers.TextVectorization(
            standardize=None, 
            output_mode='int', 
            output_sequence_length=MAX_LENGTH,
        )
        vectorizer.adapt(train_sentences)
        return vectorizer

    # Function to count the frequency of words.
    def count_word_frequencies(self, dataset, vectorizer):
        word_counts = {}

        for text in dataset:
            # Convert tokenized text into words.
            words = self.get_words_from_indices(text.numpy(), vectorizer)

            # Delete the word [UNK].
            words = [word for word in words if word not in ['[UNK]', '[OOV]']]

            # Count the frequency of each word.
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1

        # Create a table with the words and their frequencies.
        word_table = pd.DataFrame(list(word_counts.items()), columns=['Word', 'Frequency'])
        word_table = word_table.sort_values(by='Frequency', ascending=False)

        return word_table

    # Function to revert indexes to words.
    def get_words_from_indices(self, indices, vectorizer):
        # Get the vectorizer vocabulary.
        vocabulary = vectorizer.get_vocabulary(include_special_tokens=True)
        
        # Convert indexes to words.
        words = [vocabulary[index] for index in indices if index < len(vocabulary) and index > 0]
        
        return words

    # Function to create the bar graph.
    def plot_word_frequency(self, word_table, index=PLOT_INDEX):
        # Create the bar graph for the most frequent words.
        plt.figure(figsize=(10, 6))
        plt.bar(word_table['Word'][:index], word_table['Frequency'][:index], color='skyblue')
        plt.xticks(rotation=90)
        plt.xlabel('Words')
        plt.ylabel('Frecuency')
        plt.title('Most frequently used words')
        plt.show()

    # Function to generate the word cloud.
    def generate_wordcloud(self, word_table):
        word_freq_dict = dict(zip(word_table['Word'], word_table['Frequency']))

        # Create the word cloud.
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq_dict)

        # Show the word cloud graph.
        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('WordCloud')
        plt.show()

    # Function to generate the word cloud without the graphic.
    def generate_wordcloud_only(self, word_table):
        word_freq_dict = dict(zip(word_table['Word'], word_table['Frequency']))

        # Create the word cloud.
        return WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq_dict)


    # Preprocess the dataset and create the table of the most frequent words.
    def preprocess_and_create_word_table(self, dataset, vectorizer):
        # Apply preprocessing.
        dataset = dataset.map(lambda x: self.preprocess_text(x))  # Only text, without labels
        
        # Tokenize the text
        tokenized_dataset = dataset.map(lambda x: vectorizer(x))

        # To count word frequencies.
        word_table = self.count_word_frequencies(tokenized_dataset, vectorizer)

        return word_table

    # Upload the text file (set the path to your .txt file).
    def load_text_file(self, file_path):
        dataset = tf.data.TextLineDataset(file_path)
        return dataset
    
    def splitFileNameWithoutExt(self, tituloventana):

        tituloventana_local = os.path.basename(tituloventana)[:-4]
        fileNameWithoutExt = re.findall(r'[A-Z][^A-Z]*',tituloventana_local)
        fileNameWithoutExt = ' '.join(fileNameWithoutExt)

        # Set global variable FILE_NAME_WITHOUT_EXT and DIRECTORY_NAME
        global FILE_NAME_WITHOUT_EXT
        FILE_NAME_WITHOUT_EXT = tituloventana_local

        global DIRECTORY_NAME
        DIRECTORY_NAME = os.path.dirname(tituloventana)

        return fileNameWithoutExt
    
    def save_table_to_csv_png_close_window(self, event, table):
        # To save the table data as a CSV file.
        table.to_csv(f"{DIRECTORY_NAME}/{FILE_NAME_WITHOUT_EXT}.csv", index=False)
        print(f"Data stored in '{DIRECTORY_NAME}/{FILE_NAME_WITHOUT_EXT}.csv'")

       # Capture the graph image and save it to disk.
        plt.savefig(f"{DIRECTORY_NAME}/{FILE_NAME_WITHOUT_EXT}.png")  # To save the chart image as a png file.
        print(f"Chart saved as '{DIRECTORY_NAME}/{FILE_NAME_WITHOUT_EXT}.png'")

        # Close the figure window.
        plt.close()        

    
    def plot_all_in_one(self, dataset, vectorizer, index=PLOT_INDEX, tituloventana=''):
        # Create a figure and a grid of subplots using gridspec
        fig = plt.figure(figsize=(12, 8))

        # Adjust the value of PLOT_INDEX and FILE_NAME
        global PLOT_INDEX
        PLOT_INDEX = index

        # Use gridspec to control the layout
        gs = fig.add_gridspec(2, 2, width_ratios=[1, 2], height_ratios=[5, 1]) 

        # Add the table chart (on the left, occupying the full height)

        # Create the table of most frequent words
        word_table = self.preprocess_and_create_word_table(dataset, vectorizer)
        word_table.reset_index(drop=True, inplace=True)

        wordtable = word_table.head(index)
        ax1 = fig.add_subplot(gs[0, 0])  # Takes up the entire left column
        ax1.set_title('Words and Frecuency Table.', fontsize=14, loc='center', pad=20)        
        ax1.axis('off')  # No axes for the table
        table = ax1.table(cellText=wordtable.values, colLabels=wordtable.columns, loc='center', cellLoc='center')

        # Create a download button
        # Create the button below the table
        ax_button = fig.add_subplot(gs[1, 0])  # Row 1, Column 0
        ax_button.axis('off')  # Turn off the axes for the button

        # Table formatting:
        # Change header background to gray and text to white
        for (i, j), cell in table.get_celld().items():
            if i == 0:  # Table header
                cell.set_text_props(weight='bold', color='white')  # White and bold text
                cell.set_facecolor('gray')  # Gray background
            else:
                cell.set_text_props(weight='normal', color='black')  # Normal black text
                cell.set_facecolor('white')  # White background for data


        # Add the Wordcloud chart (at the top-right)
        ax2 = fig.add_subplot(gs[0, 1])  # Row 0, Column 1
        ax2.imshow(self.generate_wordcloud_only(word_table), interpolation='bilinear')
        ax2.axis('off')   # No axes for the wordcloud
        ax2.set_title('Wordcloud of Word Frequency', fontsize=14, loc='center', pad=20)

        # Create the bar chart for the most frequent words
        categorias = word_table['Word'][:index]
        ax3 = fig.add_subplot(gs[1,1])
        ax3.bar(categorias, word_table['Frequency'][:index], color='skyblue', edgecolor='black')
        ax3.set_xticklabels(categorias, rotation=45)
        ax3.set_xticks(categorias)
        ax3.set_title('Most Frequent Words', fontsize=14, loc='center', pad=20)
        ax3.set_xlabel('Words')
        ax3.set_ylabel('Frecuency')

        # Create the button below the table
        ax_button = fig.add_subplot(gs[1, 0])  # Row 1, Column 0
        ax_button.axis('off')  # Turn off the axes for the button
        button = Button(ax_button, 'Download CSV')
        
        # Customize the button: Change the background and text color
        button.ax.set_facecolor('lightblue')  # Background color
        button.label.set_color('red')  # Text color
        button.label.set_fontsize(14)  # Font size
        button.label.set_fontweight('bold')  # Bold font

        # Trigger the on-click event
        button.on_clicked(lambda event: self.save_table_to_csv_png_close_window(event, wordtable))

        # Adjust the window title
        fig_manager = plt.get_current_fig_manager()
        fig_manager.set_window_title(self.splitFileNameWithoutExt(tituloventana))

        # Adjust the layout to prevent overlaps
        plt.tight_layout()
         # Show the chart
        plt.show()