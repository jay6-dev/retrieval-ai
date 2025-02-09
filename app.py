import gradio as gr
import torch
from transformers import CLIPProcessor, CLIPModel
import numpy as np
import kagglehub
from PIL import Image
import os
from pathlib import Path
import logging
import faiss
from tqdm import tqdm
import speech_recognition as sr
from gtts import gTTS
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class ImageSearchSystem:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Load CLIP model
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(self.device)
        
        # Initialize dataset
        self.image_paths = []
        self.index = None
        self.initialized = False
        
    def initialize_dataset(self) -> None:
        """Download and process dataset"""
        try:
            path = kagglehub.dataset_download("alessandrasala79/ai-vs-human-generated-dataset")
            image_folder = os.path.join(path, 'test_data_v2')
            
            self.image_paths = [
                f for f in Path(image_folder).glob("**/*") 
                if f.suffix.lower() in ['.jpg', '.jpeg', '.png']
            ]
            
            if not self.image_paths:
                raise ValueError(f"No images found in {image_folder}")
            
            logger.info(f"Found {len(self.image_paths)} images")
            
            self._create_image_index()
            self.initialized = True
            
        except Exception as e:
            logger.error(f"Dataset initialization failed: {str(e)}")
            raise

    def _create_image_index(self, batch_size: int = 32) -> None:
        """Create FAISS index"""
        try:
            all_features = []
            
            for i in tqdm(range(0, len(self.image_paths), batch_size), desc="Indexing images"):
                batch_paths = self.image_paths[i:i + batch_size]
                batch_images = [Image.open(img).convert("RGB") for img in batch_paths]

                if batch_images:
                    inputs = self.processor(images=batch_images, return_tensors="pt", padding=True)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        features = self.model.get_image_features(**inputs)
                        features = features / features.norm(dim=-1, keepdim=True)
                        all_features.append(features.cpu().numpy())
            
            all_features = np.concatenate(all_features, axis=0)
            self.index = faiss.IndexFlatIP(all_features.shape[1])
            self.index.add(all_features)
            
            logger.info("Image index created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create image index: {str(e)}")
            raise

    def search(self, query: str, audio_path: str = None, k: int = 5):
        """Search for images using text or speech"""
        try:
            if not self.initialized:
                raise RuntimeError("System not initialized. Call initialize_dataset() first.")
            
            # Convert speech to text if audio input is provided
            if audio_path:
                recognizer = sr.Recognizer()
                with sr.AudioFile(audio_path) as source:
                    audio_data = recognizer.record(source)
                    try:
                        query = recognizer.recognize_google(audio_data)
                    except sr.UnknownValueError:
                        return [], "Could not understand the spoken query.", None

            # Process text query
            inputs = self.processor(text=[query], return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Search FAISS index
            scores, indices = self.index.search(text_features.cpu().numpy(), k)
            results = [Image.open(self.image_paths[idx]) for idx in indices[0]]

            # Generate Text-to-Speech
            tts = gTTS(f"Showing results for {query}")
            temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            tts.save(temp_audio.name)

            return results, query, temp_audio.name

        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return [], "Error during search.", None

def create_demo_interface() -> gr.Interface:
    """Create Gradio interface with dark mode & speech support"""
    system = ImageSearchSystem()
    
    try:
        system.initialize_dataset()
    except Exception as e:
        logger.error(f"Failed to initialize system: {str(e)}")
        raise
    
    examples = [
        ["a beautiful landscape with mountains"],
        ["people working in an office"],
        ["a cute dog playing"],
        ["a modern city skyline at night"],
        ["a delicious-looking meal"]
    ]
    
    return gr.Interface(
        fn=system.search,
        inputs=[
            gr.Textbox(label="Enter your search query:", placeholder="Describe the image...", lines=2),
            gr.Audio(source="microphone", type="filepath", label="Speak Your Query (Optional)")
        ],
        outputs=[
            gr.Gallery(label="Search Results", show_label=True, columns=5, height="auto"),
            gr.Textbox(label="Spoken Query", interactive=False),
            gr.Audio(label="Results Spoken Out Loud")
        ],
        title="Multi-Modal Image Search",
        description="Use text or voice to search for images.",
        theme="dark",
        examples=examples,
        cache_examples=True,
        css=".gradio-container {background-color: #121212; color: #ffffff;}"
    )

if __name__ == "__main__":
    try:
        demo = create_demo_interface()
        demo.launch(share=True, enable_queue=True, max_threads=40)
    except Exception as e:
        logger.error(f"Failed to launch app: {str(e)}")
        raise
