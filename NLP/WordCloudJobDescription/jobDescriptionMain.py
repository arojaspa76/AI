from jobDescriptionWordCloudClasess import jobDescriptionOperations
from jobDescriptionExceptions import ErrorReadingFile

import tkinter as tk
from tkinter import filedialog

import argparse

def main(language, total):
    """
    Instantiates the main function. This script to determine the most frequently used words in a job posting description.
    
    Args:
        language: Language used to get the stopwords.
        total: Total amount of words to analyze
    
    Returns:
        csv_file: This file containg the most frequently used words in a job posting description
        png_file: This pic show the table, wordclod and a bar graph with the most frequently used words in a job posting description
    """   
    
    try:
        jdop = jobDescriptionOperations(language)

        # Create the main Tkinter window (it will not be displayed)
        root = tk.Tk()
        root.withdraw()  # Hide the main window

        # Display the dialog box for selecting a file.
        file_path = filedialog.askopenfilename(title="Select a file with the Job Description to be analyzed", filetypes=[("Text Files", "*.txt")])

        # Show the path of the selected file
        if file_path:
            jobDescriptionFile = file_path
            print("Selected file:", file_path)
        else:
            raise ErrorReadingFile("No file selected.")

        dataset = jdop.load_text_file(jobDescriptionFile)

        # To create and customize the vectorizer.
        vectorizer = jdop.create_vectorizer(dataset)

        jdop.plot_all_in_one(dataset, vectorizer, total, jobDescriptionFile)
    
    except ErrorReadingFile as e:
        print(f"An error has occurred: {e}")

if __name__ == '__main__':

    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Arguments to setup the wordcloud information")
    
    # Define command-line arguments
    parser.add_argument('-l', '--lan', type=str, default='en', help="Language to get stop words", required=True)
    parser.add_argument('-t', '--total', type=int, default=20, help="Total amount of words to analyze", required=False)    

    # Parse the arguments
    args = parser.parse_args()

    main(args.lan, args.total)