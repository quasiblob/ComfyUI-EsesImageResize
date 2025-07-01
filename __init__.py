from .image_resize import EsesImageResize


# --- Node Registration ---

NODE_CLASS_MAPPINGS = {
    "EsesImageResize": EsesImageResize
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EsesImageResize": "Eses Image Resize"
}