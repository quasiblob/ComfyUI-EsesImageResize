# ==========================================================================
# Eses Image Resize
# ==========================================================================
#
# Description:
# The 'Eses Image Resize' node provides comprehensive image resizing
# capabilities within ComfyUI. It supports various scaling modes including
# scaling by a specific ratio, target megapixels, or directly to fixed
# dimensions. The node offers robust framing options to handle aspect
# ratio changes, allowing users to 'Crop to Fit' (fill) the target frame
# or 'Fit to Frame' (letterbox) the image with a customizable fill color.
# It also generates and outputs a corresponding mask, with control over
# the letterbox area's color (black or white) within the mask.
#
# Key Features:
#
# • Multiple Scaling Modes:
#   • `ratio`: Resizes by a simple multiplication factor.
#   • `megapixels`: Scales the image to a target megapixel count.
#   • `megapixels_with_ar`: Scales to target megapixels while maintaining
#     a specific output aspect ratio (width:height).
#   • `dimensions`: Resizes to exact width and height.
#
# • Aspect Ratio Handling:
#   • `crop_to_fit`: Resizes and then crops the image to perfectly fill
#     the target dimensions, preserving aspect ratio by removing excess.
#   • `fit_to_frame`: Resizes and adds a letterbox/pillarbox to fit the
#     image within the target dimensions without cropping, filling empty
#     space with a specified color.
#
# • Customizable Fill Color:
#   • `letterbox_fill_color`: Sets the RGB color for the letterbox/
#     pillarbox areas when 'Fit to Frame' is active. Supports RGB/RGBA and
#     hex color codes.
#
# • Mask Output Control:
#   • Automatically generates a mask corresponding to the resized image.
#   • `letterbox_mask_is_white`: Determines if the letterbox areas in the
#     output mask should be white (active) or black (inactive).
#
# • Advanced Filters:
#   • `resample_filter`: Selects the resampling algorithm (bicubic,
#     bilinear, nearest) for image quality.
#   • `sharpen_output_image`: Applies a sharpening filter after resizing.
#
# How to Use (Examples):
# • Resize to 1024x1024, cropping excess: Set `scale_mode` to "dimensions",
#   `target_width` and `target_height` to 1024, and `crop_to_fit` to True.
#
# • Scale image to 1MP, maintaining 16:9 aspect ratio, letterboxing: Set
#   `scale_mode` to "megapixels_with_ar", `target_megapixels` to 1.0,
#   `ar_width` to 16, `ar_height` to 9, and `fit_to_frame` to True.
#
# • Change letterbox color: Use `letterbox_fill_color` (e.g., "255,0,0" for
#   red or "#00FF00" for green).
#
# Node Outputs:
# • `IMAGE`: The resized image tensor.
#
#  • `MASK`: The corresponding mask tensor, reflecting cropping or letterbox
#   areas.
#
# • `INFO`: A string providing details about the resizing operation,
#   including final dimensions, megapixels, aspect ratio, and framing
#   strategy.
#
# Version: 1.2.1
#
# License: See LICENSE.txt
#
# ==========================================================================

import torch
from PIL import Image, ImageFilter
import numpy as np
import math
import comfy.utils # type: ignore
from comfy import model_management # type: ignore



# --- Helper Functions ---

# This function uses comfy's 
# internal features and is very
# similar to nodes_upscale_model.py
# it basically calls tiled_scale, with 
# the same boilerplate setup and runs 
# if it has memory. It tries drop to 
# smaller tile size, if out of memory 
# is encountered
def _apply_model_upscaling(image_tensor, upscale_model):
    """Applies the upscale model to the image tensor using tiled, auto-retrying processing."""
    if upscale_model is None:
        return image_tensor

    device = model_management.get_torch_device()

    # Pre-allocate memory on the target device
    memory_required = model_management.module_size(upscale_model.model) + image_tensor.nelement() * image_tensor.element_size()
    model_management.free_memory(memory_required, device)
    upscale_model.to(device)
    
    # Convert image from ComfyUI's [B, H, W, C] 
    # to PyTorch's [B, C, H, W]
    pytorch_image = image_tensor.movedim(-1, -3).to(device)

    # Start with a large tile 
    # size for efficiency.
    tile_size = 512 
    overlap = 32
    
    # Try upscaling up to 3 times, 
    # reducing tile size on each attempt.
    for attempt in range(3):
        try:
            # Calculate total steps for the progress bar.
            total_steps = pytorch_image.shape[0] * comfy.utils.get_tiled_scale_steps(
                pytorch_image.shape[3], 
                pytorch_image.shape[2], 
                tile_x=tile_size, 
                tile_y=tile_size, 
                overlap=overlap
            )

            progress_bar = comfy.utils.ProgressBar(total_steps)
            
            # Perform the tiled upscaling.
            upscaled_tensor = comfy.utils.tiled_scale(
                pytorch_image,
                lambda image_tile: upscale_model(image_tile),
                tile_x=tile_size,
                tile_y=tile_size,
                overlap=overlap,
                upscale_amount=upscale_model.scale,
                pbar=progress_bar
            )
            
            # If successful, move the model to 
            # CPU and prepare the final tensor.
            upscale_model.to("cpu")
            final_tensor = torch.clamp(upscaled_tensor.movedim(-3, -1), min=0.0, max=1.0)
            return final_tensor

        except model_management.OOM_EXCEPTION as oom_exception:
            print(f"Warning: Out of memory during upscaling on attempt {attempt + 1}. Halving tile size to {tile_size // 2}.")
            tile_size //= 2

            # If tile size becomes too small, 
            # we must raise the exception.
            if tile_size < 128:
                # Ensure model is moved off GPU before failing
                upscale_model.to("cpu") 
                raise oom_exception
    
    # If all attempts fail, move model to
    # CPU and raise the last exception.
    upscale_model.to("cpu")
    raise RuntimeError("Failed to upscale image after multiple attempts due to OOM errors.")


def comfy_image_to_pil(image_tensor):
    """Converts a ComfyUI image tensor to a PIL Image."""
    
    # image_tensor is BATCH x HEIGHT x WIDTH x CHANNEL
    batch_size, H, W, C = image_tensor.shape
    
    if batch_size > 1:
        # Warning for batch size > 1 as this node processes one image at a time
        print("Warning: Input image batch size > 1. Processing only the first image in the batch.")

    img_np = image_tensor[0].cpu().numpy()
    img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
    
    return Image.fromarray(img_np)


def pil_to_comfy_image(pil_image):
    """Converts a PIL Image to a ComfyUI image tensor."""
    img_np = np.array(pil_image).astype(np.float32) / 255.0
    # Add batch dimension [1, H, W, C]
    img_tensor = torch.from_numpy(img_np)[None,]
    
    return img_tensor


def comfy_mask_to_pil(mask_tensor):
    """Converts a ComfyUI mask tensor to a PIL Image (grayscale)."""
    # Mask is BATCH x HEIGHT x WIDTH
    batch_size, H, W = mask_tensor.shape
    
    if batch_size > 1:
        # Warning for batch size > 1 as this node processes one mask at a time
        print("Warning: Input mask batch size > 1. Processing only the first mask in the batch.")
    
    mask_np = mask_tensor[0].cpu().numpy()
    img_np = np.clip(mask_np * 255, 0, 255).astype(np.uint8)
    
    return Image.fromarray(img_np, mode='L') # 'L' for grayscale


def pil_to_comfy_mask(pil_mask):
    """Converts a PIL Image (grayscale) to a ComfyUI mask tensor."""
    mask_np = np.array(pil_mask).astype(np.float32) / 255.0
    
    # Add batch dimension [1, H, W]
    mask_tensor = torch.from_numpy(mask_np)[None,]
    return mask_tensor


def _parse_color_string(color_string):
    """
    Parses a color string (Hex: RRGGBB or RRGGBBAA; or RGB/RGBA: "R,G,B" or "R,G,B,A").
    Returns an (R, G, B, A) tuple (0-255). Defaults to opaque black if parsing fails.
    """
    color_string = color_string.strip()
    
    # Try parsing as Hex (e.g., "RRGGBB" or "RRGGBBAA")
    try:
        if len(color_string) == 6:
            return tuple(int(color_string[i:i+2], 16) for i in (0, 2, 4)) + (255,) # Add full alpha
        elif len(color_string) == 8:
            return tuple(int(color_string[i:i+2], 16) for i in (0, 2, 4, 6))
    except ValueError:
        pass # Not a valid hex string, try RGB

    # Try parsing as RGB comma-separated (e.g., "255,128,0" or "255,128,0,255")
    try:
        parts = [int(p.strip()) for p in color_string.split(',')]
        if len(parts) == 3:
            return tuple(parts) + (255,) # Add full alpha
        elif len(parts) == 4:
            return tuple(parts)
    except ValueError:
        pass # Not a valid RGB string

    print(f"Warning: Could not parse color string '{color_string}'. Defaulting to opaque black (0,0,0,255).")
    return (0, 0, 0, 255) # Fallback to opaque black for robust letterbox fill


def _calculate_initial_target_dimensions(original_width, original_height, scale_mode, 
                                        multiplier, megapixels, target_width, target_height,
                                        ar_width, ar_height, keep_aspect_ratio):
    """Calculates the target dimensions based on the chosen scale mode."""
    new_width, new_height = original_width, original_height
    original_aspect_ratio = original_width / original_height if original_height != 0 else 1.0

    if scale_mode == "multiplier":
        new_width = round(original_width * multiplier)
        new_height = round(original_height * multiplier)

    elif scale_mode == "megapixels":
        target_pixels = megapixels * 1_000_000
        new_height = round(np.sqrt(target_pixels / original_aspect_ratio))
        new_width = round(new_height * original_aspect_ratio)

    elif scale_mode == "target_width":
        new_width = target_width
        if keep_aspect_ratio:
            new_height = round(target_width / original_aspect_ratio)
        else:
            new_height = original_height
        
    elif scale_mode == "target_height":
        new_height = target_height
        if keep_aspect_ratio:
            new_width = round(target_height * original_aspect_ratio)
        else:
            new_width = original_width

    elif scale_mode == "both_dimensions":
        new_width = target_width
        new_height = target_height
        
    elif scale_mode == "megapixels_with_ar":
        target_pixels = megapixels * 1_000_000
        # Handle zero aspect ratio components to prevent division by zero
        target_ar = ar_width / ar_height if ar_height != 0 else (ar_width if ar_width != 0 else 1.0)
        
        new_height = round(np.sqrt(target_pixels / target_ar))
        new_width = round(new_height * target_ar)
    
    # Ensure calculated dimensions are at least 1 pixel
    new_width = max(1, round(new_width))
    new_height = max(1, round(new_height))

    return new_width, new_height


def _apply_divisible_by_rounding(width, height, divisible_by):
    """Applies rounding to ensure dimensions are divisible by the specified number."""
    if divisible_by > 0:
        width = round(width / divisible_by) * divisible_by
        height = round(height / divisible_by) * divisible_by
        
        # Ensure dimensions are never 0 or negative after rounding
        width = max(1, width)
        height = max(1, height)

    return int(width), int(height) # Ensure integer output


def _perform_scaling_and_cropping(pil_image, pil_mask, final_width, final_height, resample_filter, crop_to_fit, fit_to_frame, letterbox_fill_color_rgb, letterbox_mask_is_white, scale_mode):
    """
    Performs the actual image scaling and optional cropping/padding.
    Handles 'Crop to Fit' (fill), 'Fit to Frame' (contain/letterbox), or default scaling.
    """
    
    original_width, original_height = pil_image.size
    original_aspect_ratio = original_width / original_height if original_height != 0 else 1.0
    
    # Modes where special framing logic (crop_to_fit, fit_to_frame) 
    # can apply to define a canvas.
    # megapixels_with_ar is now included here, 
    # allowing both crop_to_fit and fit_to_frame for it.
    apply_canvas_framing_logic = scale_mode in ["target_width", "target_height", "both_dimensions", "megapixels_with_ar"]

    resized_image = None
    resized_mask = None

    # Priority: Crop to Fit > Fit to Frame > Default Scaling
    if crop_to_fit and apply_canvas_framing_logic:
        
        # --- Crop to Fit (Fill) Logic ---
        
        # Scales the image to completely fill the 
        # target dimensions, then crops any excess.
        target_aspect_ratio = final_width / final_height if final_height != 0 else (final_width if final_width != 0 else 1.0)

        scale_factor = 1.0
        if original_aspect_ratio > target_aspect_ratio:
            # Original is wider than target: scale by 
            # height to fill, then crop width
            scale_factor = final_height / original_height
        else:
            # Original is taller or same AR as target: 
            # scale by width to fill, then crop height
            scale_factor = final_width / original_width
        
        intermediate_width = int(original_width * scale_factor)
        intermediate_height = int(original_height * scale_factor)

        # Ensure intermediate dimensions are at least 1 pixel
        intermediate_width = max(1, intermediate_width)
        intermediate_height = max(1, intermediate_height)

        # 1. Scale the image to fill the target 
        # dimensions while preserving its original AR
        scaled_img_for_crop = pil_image.resize((intermediate_width, intermediate_height), resample=resample_filter)
        scaled_mask_for_crop = None
        
        if pil_mask is not None:
            scaled_mask_for_crop = pil_mask.resize((intermediate_width, intermediate_height), resample=Image.Resampling.NEAREST)

        # 2. Calculate crop box for a center crop
        left = (intermediate_width - final_width) / 2
        top = (intermediate_height - final_height) / 2
        right = (intermediate_width + final_width) / 2
        bottom = (intermediate_height + final_height) / 2
        
        crop_box = (math.floor(left), math.floor(top), math.ceil(right), math.ceil(bottom))

        # 3. Perform the crop
        resized_image = scaled_img_for_crop.crop(crop_box)
        
        if scaled_mask_for_crop is not None:
            resized_mask = scaled_mask_for_crop.crop(crop_box)

    # fit_to_frame now works with 
    # megapixels_with_ar as well
    elif fit_to_frame and apply_canvas_framing_logic:
        
        # --- Fit to Frame (Contain/Letterbox) Logic ---
        
        # Scales the image to fit entirely within 
        # the target dimensions, then pads with bars.
        target_aspect_ratio = final_width / final_height if final_height != 0 else (final_width if final_width != 0 else 1.0)
        scale_factor = 1.0
        
        if original_aspect_ratio > target_aspect_ratio:
            
            # Original is wider than target: scale by 
            # width to fit, height will be smaller
            scale_factor = final_width / original_width
        else:
            # Original is taller or same AR as target: 
            # scale by height to fit, width will be smaller
            scale_factor = final_height / original_height

        scaled_width = int(original_width * scale_factor)
        scaled_height = int(original_height * scale_factor)


        # Ensure scaled dimensions are at least 1 pixel
        scaled_width = max(1, scaled_width)
        scaled_height = max(1, scaled_height)

        # 1. Create a new background canvas of the 
        # final target size with the specified color
        resized_image = Image.new('RGB', (final_width, final_height), letterbox_fill_color_rgb)
        
        # Determine mask padding color 
        # (0 for black, 255 for white)
        mask_padding_color = 255 if letterbox_mask_is_white else 0
        
        if pil_mask is not None:
            resized_mask = Image.new('L', (final_width, final_height), mask_padding_color) 

        # 2. Resize the original image/mask 
        # to fit within the target frame
        scaled_original_image = pil_image.resize((scaled_width, scaled_height), resample=resample_filter)
        scaled_original_mask = None
        
        if pil_mask is not None:
            scaled_original_mask = pil_mask.resize((scaled_width, scaled_height), resample=Image.Resampling.NEAREST)

        # 3. Calculate paste position to center 
        # the scaled image/mask on the new canvas
        paste_x = (final_width - scaled_width) // 2
        paste_y = (final_height - scaled_height) // 2

        # 4. Paste the scaled image/mask onto the new canvas
        resized_image.paste(scaled_original_image, (paste_x, paste_y))
        
        if pil_mask is not None:
            resized_mask.paste(scaled_original_mask, (paste_x, paste_y))

    else:
        
        # --- Default Scaling Logic ---
        # Resizes directly to final_width/height. 
        # Aspect ratio is preserved if 'keep_aspect_ratio' is True
        # in the dimension calculation step; otherwise, 
        # distortion may occur.
        resized_image = pil_image.resize((final_width, final_height), resample=resample_filter)
        
        if pil_mask is not None:
            resized_mask = pil_mask.resize((final_width, final_height), resample=Image.Resampling.NEAREST)
        else:
            resized_mask = None # No mask to resize

    return resized_image, resized_mask



# --- Eses Image Resize Node Class ---

class EsesImageResize:
    def __init__(self):
        pass

    CATEGORY = "Eses Nodes/Image"
    FUNCTION = "resize_image_advanced"
    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "STRING",)
    RETURN_NAMES = ("IMAGE", "MASK", "width", "height", "info",)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", {"image_upload": True}),
                "scale_mode": (["multiplier", "megapixels", "target_width", "target_height", "both_dimensions", "megapixels_with_ar"],),
                "interpolation_method": (["area", "bilinear", "bicubic", "lanczos", "nearest-neighbor"],),
            },

            "optional": {
                "upscale_model": ("UPSCALE_MODEL",),
                "mask": ("MASK", {"optional": True}), # Optional mask input
                "ref_image": ("IMAGE", {"optional": True, "tooltip": "If connected, use this image's dimensions as the target width and height."}), # MODIFICATION: Reference Image
                "ref_mask": ("MASK", {"optional": True, "tooltip": "If connected, use this mask's dimensions as the target. Overridden by Reference Image."}), # MODIFICATION: Reference Mask
                "multiplier": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 100.0, "step": 0.01, "round": 0.001, "tooltip": "Multiplies original dimensions by this factor (Scale Mode: ratio)"}),
                "megapixels": ("FLOAT", {"default": 2.0, "min": 0.01, "max": 100.0, "step": 0.01, "round": 0.001, "tooltip": "Sets target total pixels in megapixels (Scale Mode: megapixels/megapixels_with_ar)"}),
                "target_width": ("INT", {"default": 512, "min": 8, "max": 8192, "step": 1, "tooltip": "Target width in pixels (Scale Mode: target_width/both_dimensions)"}),
                "target_height": ("INT", {"default": 512, "min": 8, "max": 8192, "step": 1, "tooltip": "Target height in pixels (Scale Mode: target_height/both_dimensions)"}),
                "ar_width": ("INT", {"default": 16, "min": 1, "max": 4096, "step": 1, "tooltip": "Aspect ratio width component (Scale Mode: megapixels_with_ar)"}),
                "ar_height": ("INT", {"default": 9, "min": 1, "max": 4096, "step": 1, "tooltip": "Aspect ratio height component (Scale Mode: megapixels_with_ar)"}),
                "keep_aspect_ratio": ("BOOLEAN", {"default": True, "tooltip": "If true, preserves original aspect ratio when using target_width/height modes (otherwise may distort)"}),
                "crop_to_fit": ("BOOLEAN", {"default": False, "tooltip": "Scales and crops the image to fill the target dimensions (no letterboxing). Takes priority over 'Fit to Frame'."}),
                "fit_to_frame": ("BOOLEAN", {"default": False, "tooltip": "Scales the image to fit entirely within the target dimensions, adding colored bars (letterboxing). Overridden by 'Crop to Fit'."}),
                "letterbox_color": ("STRING", {"default": "0,0,0", "multiline": False, "tooltip": "Color for letterboxing/padding (Hex: RRGGBB, RRGGBBAA; or RGB/RGBA: 255,255,255,255). Default: Black."}),
                "letterbox_mask_is_white": ("BOOLEAN", {"default": False, "tooltip": "If 'Fit to Frame' is active, sets the padded area in the output mask to white (255); otherwise, it's black (0)."}),
                "divisible_by": ("INT", {"default": 8, "min": 0, "max": 64, "step": 1, "tooltip": "Rounds final dimensions to be divisible by this number. Set to 0 to disable."}),
            }
        }
    

    # Main function where the node's logic resides
    # Main function where the node's logic resides
    def resize_image_advanced(self, image, scale_mode, interpolation_method,
                              upscale_model=None,
                              mask=None, ref_image=None, ref_mask=None,
                              multiplier=1.0, megapixels=2.0,
                              target_width=512, target_height=512,
                              ar_width=16, ar_height=9, 
                              keep_aspect_ratio=True, crop_to_fit=False, fit_to_frame=False, 
                              letterbox_color="0,0,0", letterbox_mask_is_white=False, 
                              divisible_by=8):

        # 1. Convert ComfyUI tensors to PIL Images for processing
        pil_image = comfy_image_to_pil(image)
        original_width, original_height = pil_image.size
        pil_mask = None
        
        if mask is not None:
            pil_mask = comfy_mask_to_pil(mask)
            
            # Ensure mask dimensions match image 
            # dimensions if a mask is provided
            if pil_mask.size != (original_width, original_height):
                print(f"Warning: Input mask dimensions {pil_mask.size} do not match image dimensions {pil_image.size}. Resizing mask to image size using NEAREST resampling.")
                pil_mask = pil_mask.resize((original_width, original_height), Image.Resampling.NEAREST)

        
        # ==================================================================
        
        # Check for Reference Image/Mask ---
        # check for reference inputs and overrides 
        # target dimensions if they exist.
        ref_width, ref_height = None, None
        
        # Prioritize ref_image over ref_mask if both are connected
        if ref_image is not None:
            # Image tensor shape is B x H x W x C
            ref_height = ref_image.shape[1]
            ref_width = ref_image.shape[2]
            print(f"Using Reference Image dimensions: {ref_width}x{ref_height}")

        elif ref_mask is not None:
            # Mask tensor shape is B x H x W
            ref_height = ref_mask.shape[1]
            ref_width = ref_mask.shape[2]
            print(f"Using Reference Mask dimensions: {ref_width}x{ref_height}")

        # If reference dimensions were found, 
        # override the target_width and target_height
        if ref_width is not None and ref_height is not None:
            target_width = ref_width
            target_height = ref_height
        # ==================================================================


        # 2. Determine the PIL resampling 
        # filter based on user input
        resample_filter = Image.Resampling.LANCZOS
        
        if interpolation_method == "nearest-neighbor":
            resample_filter = Image.Resampling.NEAREST
        elif interpolation_method == "bilinear":
            resample_filter = Image.Resampling.BILINEAR
        elif interpolation_method == "bicubic":
            resample_filter = Image.Resampling.BICUBIC
        elif interpolation_method == "lanczos":
            resample_filter = Image.Resampling.LANCZOS
        elif interpolation_method == "area":
            resample_filter = Image.Resampling.BOX 

        # 3. Parse the custom letterbox color string into an RGB tuple
        parsed_letterbox_color_rgba = _parse_color_string(letterbox_color)
        letterbox_fill_color_rgb = parsed_letterbox_color_rgba[:3] # Extract RGB components for Image.new()


        # 4. Calculate the initial target dimensions based on the chosen scaling mode
        initial_target_width, initial_target_height = _calculate_initial_target_dimensions(
            original_width, original_height, scale_mode,
            multiplier, megapixels, target_width, target_height,
            ar_width, ar_height, keep_aspect_ratio
        )

        # 5. Apply divisibility rounding to the target dimensions
        final_width, final_height = _apply_divisible_by_rounding(
            initial_target_width, initial_target_height, divisible_by
        )



        # ==================================================================
        # --- NEW: Iterative Model Upscaling Logic ---
        # ==================================================================
        
        # Check if we should use the model: must be provided and target size must be larger
        is_upscaling = final_width > original_width or final_height > original_height
        
        image_for_final_resize_tensor = image # Start with the original tensor
        
        if upscale_model is not None and is_upscaling:
            print(f"Eses-ImageResize: Model upscaling enabled. Model scale: {upscale_model.scale}x")
            
            # Loop heuristic: upscale until the next step would overshoot the target
            while True:
                current_h, current_w = image_for_final_resize_tensor.shape[1:3]
                next_w = current_w * upscale_model.scale
                next_h = current_h * upscale_model.scale

                # Stop if the next upscale would significantly exceed the target size.
                if next_w > final_width and next_h > final_height:
                    print(f"Eses-ImageResize: Stopping model upscaling at {current_w}x{current_h} to avoid overshooting target {final_width}x{final_height}.")
                    break

                # Break if the image is already larger than the target
                if current_w >= final_width and current_h >= final_height:
                    break
                
                print(f"Eses-ImageResize: Applying model upscale: {current_w}x{current_h} -> {int(next_w)}x{int(next_h)}")
                image_for_final_resize_tensor = _apply_model_upscaling(image_for_final_resize_tensor, upscale_model)

            # Convert the upscaled tensor back to a PIL image for the final step
            pil_image = comfy_image_to_pil(image_for_final_resize_tensor)
            print(f"Eses-ImageResize: Finished model upscaling. New source size: {pil_image.width}x{pil_image.height}")
            # Also update the mask to match this new size if it exists
            if mask is not None:
                 pil_mask = pil_mask.resize(pil_image.size, Image.Resampling.NEAREST)

        # ==================================================================
        # --- Resume existing logic on the (potentially upscaled) image ---
        # ==================================================================



        # 6. Perform the actual image scaling, cropping, or padding based on chosen options
        resized_image, resized_mask = _perform_scaling_and_cropping(
            pil_image, pil_mask, final_width, final_height, resample_filter,
            crop_to_fit, fit_to_frame, letterbox_fill_color_rgb, letterbox_mask_is_white, scale_mode
        )
        
        # 7. Convert processed PIL Images back to ComfyUI tensors
        output_image_tensor = pil_to_comfy_image(resized_image)
        
        # If no mask was originally input, provide a 
        # default black mask of the new dimensions
        output_mask_tensor = pil_to_comfy_mask(resized_mask) if resized_mask else torch.zeros([1, final_height, final_width], dtype=torch.float32)


        # 8. Generate an informational string 
        # about the resizing operation
        final_megapixels = (final_width * final_height) / 1_000_000.0
        final_aspect_ratio = final_width / final_height if final_height != 0 else float('inf')
        info_text = f"Resized: {final_width}x{final_height} | {final_megapixels:.2f} MP | AR: {final_aspect_ratio:.2f}"

        if scale_mode == "megapixels_with_ar":
             info_text += f" (Target AR: {ar_width}:{ar_height})"
        
        # Add details about the specific 
        # framing strategy used
        if crop_to_fit and scale_mode not in ["multiplier", "megapixels"]:
            info_text += " (Cropped to Fit)"
        elif fit_to_frame and scale_mode not in ["multiplier", "megapixels", "megapixels_with_ar"]:
            mask_color_info = "White" if letterbox_mask_is_white else "Black"
            info_text += f" (Fit to Frame - Color: {letterbox_color}, Mask: {mask_color_info})"


        # 9. Return the processed image, mask, 
        # dimensions, and info string
        return (output_image_tensor, output_mask_tensor, final_width, final_height, info_text,)
    
