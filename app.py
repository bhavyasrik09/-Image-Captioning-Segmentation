import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as T
from torchvision.models.detection import maskrcnn_resnet50_fpn
from transformers import BlipProcessor, BlipForConditionalGeneration
import numpy as np

# ====== Load Captioning Model ======
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# ====== Load Pretrained Segmentation Model ======
seg_model = maskrcnn_resnet50_fpn(pretrained=True)
seg_model.eval()

# ====== Image Transform ======
transform = T.Compose([T.ToTensor()])

# ====== Streamlit UI Config ======
st.set_page_config(page_title="Image Captioning & Segmentation", layout="wide")
st.title("🖼️ Image Captioning + 🎯 Segmentation")
st.markdown("Upload an image to generate a caption and visualize object segmentation.")

uploaded_file = st.file_uploader("📤 Upload an Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original Image", width=400)

    col1, col2 = st.columns(2)

    with st.spinner("⏳ Running captioning and segmentation..."):

        # ====== Generate Caption ======
        inputs = processor(images=image, return_tensors="pt")
        output = model.generate(**inputs)
        caption = processor.decode(output[0], skip_special_tokens=True)

        # ====== Run Segmentation ======
        img_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            pred = seg_model(img_tensor)[0]

        # ====== Draw Segmentation Masks ======
        def draw_masks(img, prediction, max_masks=5):
            img_np = np.array(img).copy()
            masks = prediction['masks']
            for i in range(min(len(masks), max_masks)):
                mask = masks[i, 0].mul(255).byte().cpu().numpy()
                red_mask = np.zeros_like(img_np)
                red_mask[:, :, 0] = mask  # Apply mask to red channel
                img_np = np.where(red_mask > 0, 0.5 * img_np + 0.5 * red_mask, img_np)
            return Image.fromarray(img_np.astype(np.uint8))

        segmented_image = draw_masks(image, pred)

    # ====== Display Output ======
    col1.subheader("📝 Caption:")
    col1.markdown(f"**`{caption}`**")

    col2.subheader("🎯 Segmented Image:")
    col2.image(segmented_image, use_column_width=True)
