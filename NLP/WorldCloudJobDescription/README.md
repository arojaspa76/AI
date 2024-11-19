# Job Description Word Cloud Analysis 🌐💼

## Project Overview 🛠️

This project aims to analyze job description files by extracting the most frequent words and displaying them using a **word cloud**, **bar chart**, and a **table** with the most frequent words. The goal is to help employers or job seekers understand the key terminology in job descriptions, which can be useful for various applications like resume optimization or recruitment.

This project uses **Python**, **TensorFlow**, **WordCloud**, and several libraries for text preprocessing and visualization. The code is organized into multiple files that allow you to preprocess text data, create a vectorizer, generate a word cloud, and visualize the results.

---

## Developed by:

**Andres Felipe Rojas Parra**  
CIO & CAIO  
Triskel Software Solutions Group  
andres.rojas@triskelss.com  
+57 (300) 5906373  
+1 (512) 5396688

---

## Features 🎉

- **Text Preprocessing**: Converts text to lowercase, removes punctuation, and filters out stopwords based on the selected language (English or Spanish).
- **Word Frequency Analysis**: Counts the frequency of words in the job description.
- **Visualization**:
  - **Word Cloud**: Displays a graphical representation of the most frequent words. 🌈
  - **Bar Chart**: Shows the frequency of the top words in a bar chart. 📊
  - **Table**: Presents the most frequent words and their frequencies in a table format. 📅
- **File Upload**: Upload job description files using a user-friendly interface. 📥
- **CSV Export**: The most frequent words can be saved in a CSV file. 💾
- **PNG Export**: The visualization can be saved as a PNG file. 🖼️

---

## Installation 🛠️

To use this project, you need to have Python and the required libraries installed. You can use the following steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/arojaspa76/AI.git
   cd AI
2. **Python version**:
    ```bash
    python --version
    python 3.12.1
3. **Create a virtual environment (optional but recommended)**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use venv\Scripts\activate
4. **Install the required dependencies**:
    ```bash
    pip install -r requirements.txt

To open the requirements.txt file, please click [here](./requirements.txt)

---

## Usage 💡
### Running the Script 🖥️

- **You can run the project using the following command**:
    ```bash
    python jobDescriptionMain.py -l <language> -t <total_words>

- **Where**:
    * -l (required) is the language of the stopwords (can be en for English or es for Spanish). 🌍
    * -t (optional) specifies the total number of words to analyze (default is 20). 🔢

- **Example**:
    ```bash
    python jobDescriptionMain.py -l en -t 30

- **This command will:**

    - Prompt you to select a text file with a job description. 📂
    - Process the job description to determine the most frequent words. 🔍
    - Display a table, word cloud, and bar chart of the most frequent words. 🖼️📊
    - Save the results to a CSV file and a PNG image. 💾🖼️

---

## Contributing 🤝

**Feel free to fork this repository, create branches, and submit pull requests with any improvements, bug fixes, or suggestions. Please make sure to**:

    1. Fork the repository. 🍴
    2. Create a new branch. 🌱
    3. Make your changes. ✨
    4. Commit your changes. 📝
    5. Open a pull request. 🔁

---

## License 📜

**This project is licensed under the MIT License - see the [LICENSE](./LICENSE.txt) file for details.**