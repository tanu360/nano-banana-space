import os
import tempfile
from PIL import Image
import gradio as gr
from io import BytesIO
from google import genai


def generate(text, images, api_key, model="gemini-2.5-flash-image-preview"):
    # Initialize client using provided api_key (or fallback to env variable)
    client = genai.Client(
        api_key=(
            api_key.strip()
            if api_key and api_key.strip() != ""
            else os.environ.get("GEMINI_API_KEY")
        )
    )

    # Prepare contents with images first, then text
    contents = images + [text]

    response = client.models.generate_content(
        model=model,
        contents=contents,
    )

    text_response = ""
    image_path = None
    for part in response.candidates[0].content.parts:
        if part.text is not None:
            text_response += part.text + "\n"
        elif part.inline_data is not None:
            # Create a temporary file to store the generated image
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                temp_path = tmp.name
                generated_image = Image.open(BytesIO(part.inline_data.data))
                generated_image.save(temp_path)
                image_path = temp_path
                print(f"Generated image saved to: {temp_path} with prompt: {text}")

    return image_path, text_response


def load_uploaded_images(uploaded_files):
    """Load and display uploaded images immediately"""
    uploaded_images = []
    if uploaded_files:
        for file in uploaded_files:
            if file.name.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                img = Image.open(file.name)
                if img.mode == "RGBA":
                    img = img.convert("RGBA")
                uploaded_images.append(img)
    return uploaded_images


def process_image_and_prompt(uploaded_files, prompt, gemini_api_key):
    try:
        input_text = prompt
        model = "gemini-2.5-flash-image-preview"

        # Load images from uploaded files
        images = []
        uploaded_images = []
        if uploaded_files:
            for file in uploaded_files:
                if file.name.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                    img = Image.open(file.name)
                    if img.mode == "RGBA":
                        img = img.convert("RGBA")
                    images.append(img)
                    uploaded_images.append(img)

        if not images:
            raise gr.Error("Please upload at least one image", duration=5)

        # Format: [dress_image, model_image, text_input] or [image1, image2, ..., text_input]
        image_path, text_response = generate(
            text=input_text, images=images, api_key=gemini_api_key, model=model
        )

        if image_path:
            # Load and convert the image if needed.
            result_img = Image.open(image_path)
            if result_img.mode == "RGBA":
                result_img = result_img.convert("RGBA")
            return (
                uploaded_images,
                [result_img],
                "",
            )  # Return uploaded images, generated image, and empty text output.
        else:
            # Return uploaded images, no generated image, and the text response.
            return uploaded_images, None, text_response
    except Exception as e:
        raise gr.Error(f"Error Getting {e}", duration=5)


# Build a Blocks-based interface with a custom HTML header and CSS
with gr.Blocks(
    css_paths="style.css",
) as demo:
    # Custom HTML header with proper class for styling
    gr.HTML(
        """
    <div class="header-container">
      <div>
          <img src="https://www.gstatic.com/lamda/images/gemini_sparkle_aurora_33f86dc0c0257da337c63.svg" alt="Gemini logo">
      </div>
      <div>
          <h1>Gemini Image Editing App</h1>
          <p>Powered by Nano-Banana</p>
      </div>
    </div>
    """
    )

    with gr.Accordion("API Configuration", open=False, elem_classes="config-accordion"):
        gr.Markdown(
            "###\n 🔑 Getting Started\n\n**Important:** This application requires a Google Gemini API key to function properly.\n\n**Steps to configure:**\n1. **Get your API key** → Visit [Google AI Studio](https://aistudio.google.com/apikey)\n2. **Create your key** → Follow the instructions to generate a new API key\n3. **Enter below** → Paste your API key in the field above (optional if set as environment variable)\n\n**Note:** The API key is processed securely and not stored permanently."
        )

    with gr.Accordion(
        "Usage Instructions", open=False, elem_classes="instructions-accordion"
    ):
        gr.Markdown(
            "###\n 📚 How to Use This Tool\n**Step-by-step guide:**\n1. 📁 **Upload images** → Select up to 10 images\n2. ✍️ **Enter your prompt** → Describe what you want to do with the images\n3. 🔑 **Add API key** → Paste your Gemini API key (if not set as environment variable)\n4. 🚀 **Generate** → Click the Generate button and wait for results\n\n**Output options:**\n- 🖼️ **Image result** → Generated/edited images will appear in the output gallery\n- 📝 **Text result** → If no image is generated, text response will appear below\n\n**Important guidelines:**\n- ✅ Use clear, descriptive prompts for better results\n- ❌ Do not upload inappropriate or NSFW content\n- 🔄 Try different prompts if results are not satisfactory"
        )

    with gr.Row(elem_classes="main-content"):
        with gr.Column(elem_classes="input-column"):
            image_input = gr.File(
                file_types=["image"],
                file_count="multiple",
                label="Select Images ",
                elem_id="image-input",
                elem_classes="upload-box",
            )
            gemini_api_key = gr.Textbox(
                lines=1,
                placeholder="Enter Gemini API Key (optional)",
                label="Gemini API Key (optional)",
                elem_classes="api-key-input",
            )
            prompt_input = gr.Textbox(
                lines=8,
                placeholder="Enter prompt here...",
                label="Prompt",
                elem_classes="prompt-input",
            )
            submit_btn = gr.Button("Start Image Editing", elem_classes="generate-btn")

        with gr.Column(elem_classes="output-column"):
            uploaded_gallery = gr.Gallery(
                label="Uploaded Images", elem_classes="uploaded-gallery"
            )
            output_gallery = gr.Gallery(
                label="Generated Outputs", elem_classes="output-gallery"
            )
            output_text = gr.Textbox(
                label="Gemini Output",
                lines=3,
                placeholder="Text response will appear here if no image is generated.",
                elem_classes="output-text",
            )

    # Set up the interaction with three outputs.
    submit_btn.click(
        fn=process_image_and_prompt,
        inputs=[image_input, prompt_input, gemini_api_key],
        outputs=[uploaded_gallery, output_gallery, output_text],
    )

    # Update uploaded gallery immediately when files are uploaded
    image_input.upload(
        fn=load_uploaded_images,
        inputs=[image_input],
        outputs=[uploaded_gallery],
    )

# demo.queue(max_size=50).launch(mcp_server=True, share=True)
demo.queue(max_size=50).launch(share=True)
