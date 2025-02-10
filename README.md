 # Retrieval-AI

## Overview
Retrieval-AI is a simple image retrieval application using a CLIP model and FAISS indexing. It allows users to search for images based on text queries and provides accessibility features such as dark mode and speech-to-text for visually impaired users.

[Retrieval - AI diagram.pdf](https://github.com/user-attachments/files/18730335/Retrieval.-.AI.diagram.pdf)

## Explanation:
1. User Inputs: The user enters a query through text or voice.
2. Gradio UI: The interface processes the input.
3. Speech-to-Text (STT): Converts voice input into text.
4. CLIP Model: Encodes the text query into embeddings.
5. FAISS Index: Searches for the most similar images.
6. Results Gallery: Displays the matched images.
7. Text-to-Speech (TTS): Reads out image descriptions for accessibility.
8. User Receives Output: Views images or listens to results.

## Features
Text-based Image Search: Users can input a query to find matching images.
Adjustable Results Count: Users can select how many results to display.
Example Queries: Predefined queries help users get started.
Dark Mode Support: Enhances usability in low-light conditions.
Speech-to-Text Input: Allows visually impaired users to speak their queries instead of typing.

## Installation
### Prerequisites
Python 3.8+
Required dependencies (see `requirements.txt` if available)

### Setup
1. Clone the repository:
   
   git clone https://github.com/yourusername/retrieval-ai.git
   cd retrieval-ai
   
2. Install dependencies:
   
   pip install -r requirements.txt
   
3. Run the application:
   
   python app.py
  
4. Open the provided URL in a browser (e.g., `http://127.0.0.1:7860`).

## Usage
1. Enter a text query or use speech input.
2. Adjust the number of results (1-10).
3. View the matched images in the results gallery.

## Accessibility Features
Dark Mode: Automatically adapts the UI for better readability in dark environments.
Speech-to-Text: Allows users to dictate their search queries for improved accessibility.

## License
This project is open-source under the MIT License.

## Contact
For any inquiries or contributions, please reach out to Joyce Nhlengetwa at jntombikayise@gmail.com.

