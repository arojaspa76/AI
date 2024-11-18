from jobDescriptionWordCloudClasess import jobDescriptionOperations
from jobDescriptionExceptions import ErrorReadingFile

import tkinter as tk
from tkinter import filedialog

import argparse

def main(language):
    """
    Instantiates the main function. This script to determine the most frequently used words in a job posting description.
    
    Args:
        language: Language used to get the stopwords.
    
    Returns:
        csv file: This file containg the most frequently used words in a job posting description
        png file: This pic show the table, wordclod and a bar graph with the most frequently used words in a job posting description
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

        jdop.plot_all_in_one(dataset, vectorizer, 20, jobDescriptionFile)
    
    except ErrorReadingFile as e:
        print(f"An error has occurred: {e}")

if __name__ == '__main__':

    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Arguments to setup the wordcloud information")
    
    # Define command-line arguments
    parser.add_argument('-l', '--lan', type=str, default='en', help="Language to get stop words", required=True)    

    # Parse the arguments
    args = parser.parse_args()

    main(args.lan)