from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch


# Load the pre-trained BLIP model
print("Loading AI Image Captioning Model...")

processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)


def generate_caption(image_path):
    """Generate a caption for an image."""

    # Open and prepare the image
    image = Image.open(image_path).convert("RGB")

    # Process the image
    inputs = processor(images=image, return_tensors="pt")

    # Generate caption
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=50
        )

    # Convert output to readable text
    caption = processor.decode(
        output[0],
        skip_special_tokens=True
    )

    return caption


# Main program
if __name__ == "__main__":

    print("\nAI Image Captioning")
    print("===================")

    image_path = input("Enter the path of the image: ")

    try:
        caption = generate_caption(image_path)

        print("\nGenerated Caption:")
        print(caption)

    except FileNotFoundError:
        print("\nError: Image file not found.")

    except Exception as e:
        print(f"\nError: {e}")
