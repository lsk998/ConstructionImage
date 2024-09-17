import streamlit as st
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
import os
from unittest.mock import patch
from transformers.dynamic_module_utils import get_imports

# Patch for the imports in transformers
def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
    if not str(filename).endswith("modeling_florence2.py"):
        return get_imports(filename)
    imports = get_imports(filename)
    imports.remove("flash_attn")
    return imports

with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
    # Load model and processor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True).to(device)
    processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)

# Function to run the model
def run_example(task_prompt, image, text_input=None):
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(generated_text, task=task_prompt, image_size=(image.width, image.height))
    return parsed_answer

# Streamlit App
st.title("Image Processing App")

# Upload an image
uploaded_image = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    image = image.convert("RGB")
    
    # Display uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Select task prompt
    task_options = ["<CAPTION>", "<DETAILED_CAPTION>", "<MORE_DETAILED_CAPTION>", "<OCR>", "<OCR_WITH_REGION>"]
    selected_task = st.selectbox("Select Task", task_options)
    
    if st.button("Run Model"):
        # Run model on the selected task
        output = run_example(selected_task, image)
        
        # Display output
        st.write("Model Output:")
        st.text(output)

