import cv2
import numpy as np
from PIL import Image
from modelscope.outputs import OutputKeys
from .utils.face_process_utils import call_face_crop, color_transfer, Face_Skin
from .utils.img_utils import img_to_tensor, tensor_to_img, tensor_to_np, np_to_tensor, np_to_mask, img_to_mask, img_to_np
from .model_holder import *
from .model_holder import load_skin_retouching_direct_models, get_skin_models

# import pydevd_pycharm
#
# pydevd_pycharm.settrace('49.7.62.197', port=10090, stdoutToServer=True, stderrToServer=True)

class RetinaFacePM:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",),
                             "multi_user_facecrop_ratio": ("FLOAT", {"default": 1, "min": 0, "max": 10, "step": 0.01}),
                             "face_index": ("INT", {"default": 0, "min": 0, "max": 10, "step": 1})
                             }}

    RETURN_TYPES = ("IMAGE", "MASK", "BOX")
    RETURN_NAMES = ("crop_image", "crop_mask", "crop_box")
    FUNCTION = "retain_face"
    CATEGORY = "protrait/model"

    def retain_face(self, image, multi_user_facecrop_ratio, face_index):
        np_image = np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
        image = Image.fromarray(np_image)
        retinaface_boxes, retinaface_keypoints, retinaface_masks, retinaface_mask_nps = call_face_crop(get_retinaface_detection(), image, multi_user_facecrop_ratio)
        crop_image = image.crop(retinaface_boxes[face_index])
        retinaface_mask = np_to_mask(retinaface_mask_nps[face_index])
        retinaface_boxe = retinaface_boxes[face_index]
        return (img_to_tensor(crop_image), retinaface_mask, retinaface_boxe)

class FaceFusionPM:

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"source_image": ("IMAGE",),
                             "swap_image": ("IMAGE",),
                             "mode": (["ali", "roop"],),
                             }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "img_face_fusion"

    CATEGORY = "protrait/model"

    def resize(self, tensor):
        image = tensor_to_img(tensor)
        short_side = max(image.width, image.height)
        resize = float(short_side / 640)
        new_size = (int(image.width // resize), int(image.height // resize))
        resize_image = image.resize(new_size, Image.Resampling.LANCZOS)
        return img_to_np(resize_image)

    def img_face_fusion(self, source_image, swap_image, mode):
        if mode == "ali":
            source_image_pil = tensor_to_img(source_image)
            swap_image_pil = tensor_to_img(swap_image)
            fusion_image = get_image_face_fusion()(dict(template=source_image_pil, user=swap_image_pil))[
                OutputKeys.OUTPUT_IMG]
            result_image = Image.fromarray(cv2.cvtColor(fusion_image, cv2.COLOR_BGR2RGB))
            return (img_to_tensor(result_image),)
        else:
            width, height = source_image.shape[2], source_image.shape[1]
            need_resize = False
            source_np = tensor_to_np(source_image)
            swap_np = tensor_to_np(swap_image)
            if source_image.shape[2] > 640 or source_image.shape[1] > 640:
                source_np = self.resize(source_image)
                need_resize = True
            if swap_image.shape[2] > 640 or swap_image.shape[1] > 640:
                swap_np = self.resize(swap_image)
            get_face_analysis().prepare(ctx_id=0, det_size=(640, 640))
            faces = get_face_analysis().get(source_np)
            swap_faces = get_face_analysis().get(swap_np)
            if len(faces) == 0:
                raise RuntimeError("No face was recognized in the source image / source image 没有识别到人脸")
            if len(swap_faces) == 0:
                print("No face was recognized in the swap faces / swap faces没有识别到人脸, 用原脸替换!!!!!!!!!")
                return (source_image,)
            result_image = get_roop().get(source_np, faces[0], swap_faces[0], paste_back=True)
            if need_resize:
                image = Image.fromarray(result_image)
                new_size = width, height
                result_image = image.resize(new_size, Image.Resampling.LANCZOS)
                result_image = img_to_np(result_image)
            return (np_to_tensor(result_image),)

class RatioMerge2ImagePM:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image1": ("IMAGE",),
                             "image2": ("IMAGE",),
                             "fusion_rate": ("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step": 0.1})
                             }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "image_ratio_merge"

    CATEGORY = "protrait/other"

    def image_ratio_merge(self, image1, image2, fusion_rate):
        rate_fusion_image = image1 * (1 - fusion_rate) + image2 * fusion_rate
        return (rate_fusion_image,)

class ReplaceBoxImgPM:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"origin_image": ("IMAGE",),
                             "box_area": ("BOX",),
                             "replace_image": ("IMAGE",),
                             }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "replace_box_image"

    CATEGORY = "protrait/model"

    def replace_box_image(self, origin_image, box_area, replace_image):
        origin_image[:, box_area[1]:box_area[3], box_area[0]:box_area[2], :] = replace_image
        return (origin_image,)

class MaskMerge2ImagePM:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image1": ("IMAGE",),
                             "image2": ("IMAGE",),
                             "mask": ("MASK",),
                             },
                }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "image_mask_merge"

    CATEGORY = "protrait/model"

    def image_mask_merge(self, image1, image2, mask, box=None):
        mask = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)
        image1 = image1 * mask + image2 * (1 - mask)
        return (image1,)

class ExpandMaskFaceWidthPM:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"mask": ("MASK",),
                             "box": ("BOX",),
                             "expand_width": ("FLOAT", {"default": 0.15, "min": 0, "max": 10, "step": 0.1})
                             }}

    RETURN_TYPES = ("MASK", "BOX")
    FUNCTION = "expand_mask_face_width"

    CATEGORY = "protrait/other"

    def expand_mask_face_width(self, mask, box, expand_width):
        h, w = mask.shape[1], mask.shape[2]

        new_mask = mask.clone().zero_()
        copy_box = np.copy(np.int32(box))

        face_width = copy_box[2] - copy_box[0]
        copy_box[0] = np.clip(np.array(copy_box[0], np.int32) - face_width * expand_width, 0, w - 1)
        copy_box[2] = np.clip(np.array(copy_box[2], np.int32) + face_width * expand_width, 0, w - 1)

        # get new input_mask
        new_mask[0, copy_box[1]:copy_box[3], copy_box[0]:copy_box[2]] = 255
        return (new_mask, copy_box)

class BoxCropImagePM:

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"image": ("IMAGE",),
                     "box": ("BOX",), }
                }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("crop_image",)
    FUNCTION = "box_crop_image"
    CATEGORY = "protrait/other"

    def box_crop_image(self, image, box):
        image = image[:, box[1]:box[3], box[0]:box[2], :]
        return (image,)

class ColorTransferPM:

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "transfer_from": ("IMAGE",),
            "transfer_to": ("IMAGE",),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "color_transfer"

    CATEGORY = "protrait/other"

    def color_transfer(self, transfer_from, transfer_to):
        transfer_result = color_transfer(tensor_to_np(transfer_from), tensor_to_np(transfer_to))  # 进行颜色迁移
        return (np_to_tensor(transfer_result),)

class FaceSkinPM:

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
            {
                "image": ("IMAGE",),
                "blur_edge": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                "blur_threshold": ("INT", {"default": 32, "min": 0, "max": 64, "step": 1}),
            },
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "face_skin_mask"

    CATEGORY = "protrait/model"

    def face_skin_mask(self, image, blur_edge, blur_threshold):
        face_skin_img = get_face_skin()(tensor_to_img(image), get_retinaface_detection(), [[1, 2, 3, 4, 5, 10, 12, 13]])[0]
        face_skin_np = img_to_np(face_skin_img)
        if blur_edge:
            face_skin_np = cv2.blur(face_skin_np, (blur_threshold, blur_threshold))
        return (np_to_mask(face_skin_np),)

class MaskDilateErodePM:

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"mask": ("MASK",), }
                }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "mask_dilate_erode"

    CATEGORY = "protrait/other"

    def mask_dilate_erode(self, mask):
        out_mask = Image.fromarray(np.uint8(cv2.dilate(tensor_to_np(mask), np.ones((96, 96), np.uint8), iterations=1) - cv2.erode(tensor_to_np(mask), np.ones((48, 48), np.uint8), iterations=1)))
        return (img_to_mask(out_mask),)

class SkinRetouchingLoaderPM:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "device": (["cuda", "cpu"], {"default": "cuda"})
        }}

    RETURN_TYPES = ("SKIN_RETOUCHING_MODEL",)
    FUNCTION = "load_skin_retouching_model"
    CATEGORY = "protrait/model"

    def load_skin_retouching_model(self, device="cuda"):
        """Load all required models for skin retouching directly"""
        import time
        print("Loading skin retouching models directly...")
        start_time = time.time()
        
        # Load all the models directly
        models = load_skin_retouching_direct_models(device)
        
        print(f"Skin retouching models loaded in {time.time() - start_time:.2f} seconds")
        return (models,)

class SkinRetouchingInferencePM:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"image": ("IMAGE",),
                     "model": ("SKIN_RETOUCHING_MODEL",),
                     "degree": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                     "smooth_border": ("BOOLEAN", {"default": True}),
                    }
                }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "skin_retouching_inference"
    CATEGORY = "protrait/model"

    def skin_retouching_inference(self, image, model, degree=1.0, smooth_border=True):
        """Perform skin retouching using the loaded models directly"""
        import torch
        import torch.nn.functional as F
        from modelscope.models.cv.skin_retouching.utils import get_roi_without_padding, roi_to_tensor
        from modelscope.models.cv.skin_retouching.utils import get_crop_bbox, preprocess_roi, smooth_border_mg
        from modelscope.outputs import OutputKeys
        import time
        
        # Extract models from the model dictionary
        generator = model["generator"]
        inpainting_net = model["inpainting_net"]
        detection_net = model["detection_net"]
        face_detector = model["face_detector"]
        diffuse_mask = model["diffuse_mask"]
        device = model.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        
        # Convert input image to numpy array
        start_time = time.time()
        input_img = tensor_to_img(image)
        
        # Check input image color channels
        input_rgb_sample = np.array(input_img)[0:3, 0:3, :]
        print(f"Input image color sample (RGB from PIL): {input_rgb_sample}")
        
        rgb_image = np.array(input_img)
        print(f"Input image shape: {rgb_image.shape}")
        print(f"Input value range: min={np.min(rgb_image)}, max={np.max(rgb_image)}")
        
        print(f"Image conversion took {time.time() - start_time:.4f} seconds")
        
        # Create output tensor
        start_time = time.time()
        output_pred = torch.from_numpy(rgb_image).to(device)
        
        # Detect faces
        detect_start = time.time()
        det_results = face_detector(rgb_image)
        print(f"Face detection took {time.time() - detect_start:.4f} seconds")
        
        # Convert detection results to standard format
        results = []
        for i in range(len(det_results['scores'])):
            info_dict = {}
            try:
                # Get bounding box
                info_dict['bbox'] = np.array(det_results['boxes'][i]).astype(np.int32).tolist()
                info_dict['score'] = det_results['scores'][i]
                
                # Handle keypoints (landmarks) with proper error checking
                if 'keypoints' in det_results and len(det_results['keypoints']) > i:
                    keypoints = det_results['keypoints'][i]
                    # Check if keypoints are already in the right format
                    if isinstance(keypoints, list) and len(keypoints) == 5 and all(len(point) == 2 for point in keypoints):
                        info_dict['landmarks'] = keypoints
                    else:
                        # Try to convert and reshape if needed
                        info_dict['landmarks'] = np.array(keypoints).astype(np.int32).reshape(5, 2).tolist()
                else:
                    # Generate default landmarks if none available
                    # Create reasonable positions from the bounding box
                    x1, y1, x2, y2 = info_dict['bbox']
                    w, h = x2 - x1, y2 - y1
                    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                    
                    # Simple default facial landmarks positioning
                    left_eye = [int(x1 + w * 0.3), int(y1 + h * 0.3)]
                    right_eye = [int(x2 - w * 0.3), int(y1 + h * 0.3)]
                    nose = [center_x, int(y1 + h * 0.5)]
                    left_mouth = [int(x1 + w * 0.3), int(y2 - h * 0.3)]
                    right_mouth = [int(x2 - w * 0.3), int(y2 - h * 0.3)]
                    
                    info_dict['landmarks'] = [left_eye, right_eye, nose, left_mouth, right_mouth]
                
                results.append(info_dict)
            except Exception as e:
                print(f"Error processing detection result {i}: {e}")
                import traceback
                traceback.print_exc()
        
        # Get crop boxes for all faces
        crop_bboxes = get_crop_bbox(results)
        
        # Check if faces were found
        face_num = len(crop_bboxes)
        if face_num == 0:
            print("No faces detected in the image!")
            return (image,)
            
        print(f"Found {face_num} faces")
        
        with torch.no_grad():
            # Process each face
            for face_idx, bbox in enumerate(crop_bboxes):
                face_start = time.time()
                
                # Get region of interest
                roi, expand, crop_tblr = get_roi_without_padding(rgb_image, bbox)
                
                # Debug color space
                print(f"ROI min/max values before conversion: {np.min(roi)}/{np.max(roi)}")
                roi_color_sample = roi[0:3, 0:3, :] if roi.shape[2] >= 3 else roi[0:3, 0:3]
                print(f"ROI color sample: {roi_color_sample}")
                
                # Original comment indicates BGR->RGB conversion, but we need to check if this is necessary
                # based on how the input was processed
                roi = roi_to_tensor(roi)
                roi = roi.to(device)
                roi = preprocess_roi(roi)
                
                # Apply local retouching if model is available
                if inpainting_net is not None and detection_net is not None:
                    local_start = time.time()
                    roi = self.retouch_local(roi, detection_net, inpainting_net)
                    print(f"Local retouching for face {face_idx} took {time.time() - local_start:.4f} seconds")
                
                # Apply main retouching
                roi_start = time.time()
                roi_output = self.predict_roi(roi, generator, diffuse_mask, degree, smooth_border, input_size=512)
                print(f"ROI prediction for face {face_idx} took {time.time() - roi_start:.4f} seconds")
                
                # Place result back into output image
                roi_pred = roi_output['pred']
                output_pred[crop_tblr[0]:crop_tblr[1], crop_tblr[2]:crop_tblr[3]] = roi_pred
                
                print(f"Face {face_idx} processing took {time.time() - face_start:.4f} seconds")
        
        # Convert to numpy array
        if not isinstance(output_pred, np.ndarray):
            output_pred = output_pred.cpu().numpy()
        
        # Debug output color
        output_sample = output_pred[0:3, 0:3, :] if len(output_pred.shape) >= 3 and output_pred.shape[2] >= 3 else output_pred[0:3, 0:3]
        print(f"Output value range: min={np.min(output_pred)}, max={np.max(output_pred)}")
        print(f"Output color sample: {output_sample}")
        
        print(f"Total processing took {time.time() - start_time:.4f} seconds")
        
        # Use np_to_tensor to properly convert back to tensor in the same color space
        result = np_to_tensor(output_pred)
        
        # Add final check of tensor values
        tensor_sample = result[0, 0:3, 0:3, :].cpu().numpy()
        print(f"Final tensor color sample: {tensor_sample}")
        
        # Return the retouched image
        return (result,)
    
    def retouch_local(self, image, detection_net, inpainting_net, patch_size=512):
        """Apply local retouching to the image"""
        import torch
        import torch.nn.functional as F
        from modelscope.models.cv.skin_retouching.utils import patch_partition_overlap, patch_aggregation_overlap
        
        with torch.no_grad():
            sub_H, sub_W = image.shape[2:]
            
            # Resize image for detection and generate mask
            sub_image_standard = F.interpolate(
                image, size=(768, 768), mode='bilinear', align_corners=True)
            sub_mask_pred = torch.sigmoid(detection_net(sub_image_standard))
            sub_mask_pred = F.interpolate(
                sub_mask_pred, size=(sub_H, sub_W), mode='nearest')
            
            # Process mask with thresholds
            sub_mask_pred_hard_low = (sub_mask_pred >= 0.35).float()
            sub_mask_pred_hard_high = (sub_mask_pred >= 0.5).float()
            sub_mask_pred = sub_mask_pred * (1 - sub_mask_pred_hard_high) + sub_mask_pred_hard_high
            sub_mask_pred = sub_mask_pred * sub_mask_pred_hard_low
            sub_mask_pred = 1 - sub_mask_pred
            
            # Standardize image sizes to multiples of patch_size
            sub_H_standard = sub_H if sub_H % patch_size == 0 else (sub_H // patch_size + 1) * patch_size
            sub_W_standard = sub_W if sub_W % patch_size == 0 else (sub_W // patch_size + 1) * patch_size
            
            # Pad images and masks
            sub_image_padding = F.pad(
                image,
                pad=(0, sub_W_standard - sub_W, 0, sub_H_standard - sub_H, 0, 0),
                mode='constant',
                value=0)
            sub_mask_pred_padding = F.pad(
                sub_mask_pred,
                pad=(0, sub_W_standard - sub_W, 0, sub_H_standard - sub_H, 0, 0),
                mode='constant',
                value=0)
            
            # Split into patches
            sub_image_padding = patch_partition_overlap(sub_image_padding, p1=patch_size, p2=patch_size)
            sub_mask_pred_padding = patch_partition_overlap(sub_mask_pred_padding, p1=patch_size, p2=patch_size)
            B_padding, C_padding, _, _ = sub_image_padding.size()
            
            # Process each patch
            sub_comp_padding_list = []
            for window_item in range(B_padding):
                sub_image_padding_window = sub_image_padding[window_item:window_item + 1]
                sub_mask_pred_padding_window = sub_mask_pred_padding[window_item:window_item + 1]
                
                # Apply mask to image
                sub_input_image_padding_window = sub_image_padding_window * sub_mask_pred_padding_window
                
                # Apply inpainting
                sub_output_padding_window = inpainting_net(
                    sub_input_image_padding_window,
                    sub_mask_pred_padding_window)
                
                # Combine original and inpainted areas
                sub_comp_padding_window = sub_input_image_padding_window + (
                    1 - sub_mask_pred_padding_window) * sub_output_padding_window
                
                sub_comp_padding_list.append(sub_comp_padding_window)
            
            # Combine all patches
            sub_comp_padding = torch.cat(sub_comp_padding_list, dim=0)
            sub_comp = patch_aggregation_overlap(
                sub_comp_padding,
                h=int(round(sub_H_standard / patch_size)),
                w=int(round(sub_W_standard / patch_size)))[:, :, :sub_H, :sub_W]
            
            return sub_comp
    
    def predict_roi(self, roi, generator, diffuse_mask, degree=1.0, smooth_border=False, return_mg=False, input_size=512):
        """Apply the main retouching to a region of interest"""
        import torch
        import torch.nn.functional as F
        from modelscope.models.cv.skin_retouching.utils import smooth_border_mg
        import numpy as np
        
        # Debug ROI input
        roi_sample = roi[0, :, 0:3, 0:3].cpu().numpy()
        print(f"ROI tensor input sample: {roi_sample}")
        
        with torch.no_grad():
            # Resize ROI to standard size
            image = F.interpolate(roi, (input_size, input_size), mode='bilinear')
            
            # Debug generator input
            gen_input_sample = image[0, :, 0:3, 0:3].cpu().numpy()
            print(f"Generator input sample: {gen_input_sample}")
            
            # Apply generator model
            pred_mg = generator(image)  # value: 0~1
            
            # Debug generator output
            gen_output_sample = pred_mg[0, :, 0:3, 0:3].cpu().numpy()
            print(f"Generator output sample: {gen_output_sample}")
            
            pred_mg = (pred_mg - 0.5) * degree + 0.5
            pred_mg = pred_mg.clamp(0.0, 1.0)
            pred_mg = F.interpolate(pred_mg, roi.shape[2:], mode='bilinear')
            pred_mg = pred_mg[0].permute(1, 2, 0)  # ndarray, (h, w, 1) or (h0, w0, 3)
            
            if len(pred_mg.shape) == 2:
                pred_mg = pred_mg[..., None]
            
            # Apply border smoothing if requested
            if smooth_border:
                pred_mg = smooth_border_mg(diffuse_mask, pred_mg)
            
            # Convert ROI to normal range
            image = (roi[0].permute(1, 2, 0) + 1.0) / 2
            
            # Debug before final blend
            roi_final_sample = image[0:3, 0:3, :].cpu().numpy()
            pred_mg_sample = pred_mg[0:3, 0:3, :].cpu().numpy()
            print(f"ROI final sample: {roi_final_sample}")
            print(f"Pred_mg sample: {pred_mg_sample}")
            
            # Apply the final retouching blend
            pred = (1 - 2 * pred_mg) * image * image + 2 * pred_mg * image  # value: 0~1
            
            # Debug blended result
            pred_sample = pred[0:3, 0:3, :].cpu().numpy()
            print(f"Blended pred sample: {pred_sample}")
            
            # Convert to byte range
            pred = (pred * 255.0).byte()  # ndarray, (h, w, 3), rgb
            
            # Debug final byte output
            pred_byte_sample = pred[0:3, 0:3, :].cpu().numpy()
            print(f"Final pred byte sample: {pred_byte_sample}")
            
            # Prepare output
            output = {'pred': pred}
            if return_mg:
                output['pred_mg'] = pred_mg.cpu().numpy()
            
            return output

class PortraitEnhancementPM:

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
            {
                "image": ("IMAGE",),
                "model": (["pgen", "real_gan"],),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "protrait_enhancement_pass"

    CATEGORY = "protrait/model"

    def protrait_enhancement_pass(self, image, model):
        if model == "pgen":
            output_image = cv2.cvtColor(get_portrait_enhancement()(tensor_to_img(image))[OutputKeys.OUTPUT_IMG], cv2.COLOR_BGR2RGB)
        elif model == "real_gan":
            output_image = cv2.cvtColor(get_real_gan_sr()(tensor_to_img(image))[OutputKeys.OUTPUT_IMG], cv2.COLOR_BGR2RGB)
        return (np_to_tensor(output_image),)

class ImageScaleShortPM:

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image": ("IMAGE",),
            "size": ("INT", {"default": 512, "min": 0, "max": 2048, "step": 1}),
            "crop_face": ("BOOLEAN", {"default": False}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "image_scale_short"

    CATEGORY = "protrait/other"

    def image_scale_short(self, image, size, crop_face):
        input_image = tensor_to_img(image)
        short_side = min(input_image.width, input_image.height)
        resize = float(short_side / size)
        new_size = (int(input_image.width // resize), int(input_image.height // resize))
        input_image = input_image.resize(new_size, Image.Resampling.LANCZOS)
        if crop_face:
            new_width = int(np.shape(input_image)[1] // 32 * 32)
            new_height = int(np.shape(input_image)[0] // 32 * 32)
            input_image = input_image.resize([new_width, new_height], Image.Resampling.LANCZOS)
        return (img_to_tensor(input_image),)

class ImageResizeTargetPM:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image": ("IMAGE",),
            "width": ("INT", {"default": 512, "min": 0, "max": 2048, "step": 1}),
            "height": ("INT", {"default": 512, "min": 0, "max": 2048, "step": 1}),
        }}

    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "image_resize_target"

    CATEGORY = "protrait/other"

    def image_resize_target(self, image, width, height):
        imagepi = tensor_to_img(image)
        out = imagepi.resize([width, height], Image.Resampling.LANCZOS)
        return (img_to_tensor(out),)

class GetImageInfoPM:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image": ("IMAGE",),
        }}

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")

    FUNCTION = "get_image_info"

    CATEGORY = "protrait/other"

    def get_image_info(self, image):
        width = image.shape[2]
        height = image.shape[1]
        return (width, height)

class MakeUpTransferPM:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "source_image": ("IMAGE",),
            "makeup_image": ("IMAGE",),
        }}

    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "makeup_transfer"

    CATEGORY = "protrait/model"

    def makeup_transfer(self, source_image, makeup_image):
        source_image = tensor_to_img(source_image)
        makeup_image = tensor_to_img(makeup_image)
        result = get_pagan_interface().transfer(source_image, makeup_image)
        return (img_to_tensor(result),)

class FaceShapMatchPM:

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "source_image": ("IMAGE",),
            "match_image": ("IMAGE",),
            "face_box": ("BOX",),
        }}

    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "faceshap_match"

    CATEGORY = "protrait/model"

    def faceshap_match(self, source_image, match_image, face_box):
        # detect face area
        source_image_copy = tensor_to_img(source_image)
        match_image_copy = tensor_to_img(match_image)
        face_skin_mask = get_face_skin()(source_image_copy, get_retinaface_detection(), needs_index=[[1, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13]])[0]
        face_width = face_box[2] - face_box[0]
        kernel_size = np.ones((int(face_width // 10), int(face_width // 10)), np.uint8)

        # Fill small holes with a close operation
        face_skin_mask = Image.fromarray(np.uint8(cv2.morphologyEx(np.array(face_skin_mask), cv2.MORPH_CLOSE, kernel_size)))

        # Use dilate to reconstruct the surrounding area of the face
        face_skin_mask = Image.fromarray(np.uint8(cv2.dilate(np.array(face_skin_mask), kernel_size, iterations=1)))
        face_skin_mask = cv2.blur(np.float32(face_skin_mask), (32, 32)) / 255

        # paste back to photo, Using I2I generation controlled solely by OpenPose, even with a very small denoise amplitude,
        # still carries the risk of introducing NSFW and global incoherence.!!! important!!!
        input_image_uint8 = np.array(source_image_copy) * face_skin_mask + np.array(match_image_copy) * (1 - face_skin_mask)

        return (np_to_tensor(input_image_uint8),)

class SuperColorTransferPM:

    @classmethod
    def INPUT_TYPES(s):
        return \
            {
                "required": {
                    "main_image": ("IMAGE",),
                    "transfer_image": ("IMAGE",),
                },
                "optional": {
                    "avatar_box": ("BOX",),
                },
            }

    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "super_color_transfer"
    CATEGORY = "protrait/super"

    def super_color_transfer(self, main_image, transfer_image, avatar_box=None):
        origin_np = tensor_to_np(main_image)
        result_np = None
        if avatar_box is not None:
            main_image = main_image[:, avatar_box[1]:avatar_box[3], avatar_box[0]:avatar_box[2], :]
            transfer_image = transfer_image[:, avatar_box[1]:avatar_box[3], avatar_box[0]:avatar_box[2], :]

        transfer_result = color_transfer(tensor_to_np(main_image), tensor_to_np(transfer_image))  # 进行颜色迁移

        face_skin_img = get_face_skin()(Image.fromarray(transfer_result), get_retinaface_detection(), [[1, 2, 3, 4, 5, 10, 12, 13]])[0]
        face_skin_np = img_to_np(face_skin_img)
        face_skin_np = cv2.blur(face_skin_np, (32, 32)) / 255

        masked_img_np = tensor_to_np(main_image) * (1 - face_skin_np) + transfer_result * face_skin_np
        result_np = masked_img_np

        if avatar_box is not None:
            origin_np[avatar_box[1]:avatar_box[3], avatar_box[0]:avatar_box[2], :] = masked_img_np
            result_np = origin_np

        return (np_to_tensor(result_np),)

class SuperMakeUpTransferPM:

    @classmethod
    def INPUT_TYPES(s):
        return \
            {
                "required": {
                    "main_image": ("IMAGE",),
                    "makeup_image": ("IMAGE",),
                },
                "optional": {
                    "avatar_box": ("BOX",),
                },
            }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "super_makeup_transfer"
    CATEGORY = "protrait/super"

    def super_makeup_transfer(self, main_image, makeup_image, avatar_box=None):
        box_width, box_height = avatar_box[2] - avatar_box[0], avatar_box[3] - avatar_box[1]
        origin_np = tensor_to_np(main_image)
        if avatar_box is not None:
            main_image = main_image[:, avatar_box[1]:avatar_box[3], avatar_box[0]:avatar_box[2], :]
            makeup_image = makeup_image[:, avatar_box[1]:avatar_box[3], avatar_box[0]:avatar_box[2], :]
        resize_source_box_image = tensor_to_img(main_image).resize([256, 256])
        resize_makeup_box_image = tensor_to_img(makeup_image).resize([256, 256])
        transfer_image = get_pagan_interface().transfer(resize_source_box_image, resize_makeup_box_image)
        box_size_transfer = transfer_image.resize([box_width, box_height], Image.Resampling.LANCZOS)
        origin_np[avatar_box[1]:avatar_box[3], avatar_box[0]:avatar_box[2], :] = img_to_np(box_size_transfer)
        return (np_to_tensor(origin_np),)

class SimilarityPM:
    @classmethod
    def INPUT_TYPES(s):
        return \
            {
                "required": {
                    "main_image": ("IMAGE",),
                    "compare_image": ("IMAGE",),
                    "model": (["sim"],),
                    "result_prefix": ("STRING", {"default": ""}),
                },
            }

    RETURN_TYPES = ("STRING",)

    FUNCTION = "similarity_compare"
    CATEGORY = "protrait/model"

    def similarity_compare(self, main_image, compare_image, model, result_prefix):
        main_image_copy = tensor_to_img(main_image)
        compare_image_copy = tensor_to_img(compare_image)
        score = None
        result = None
        if model == "sim":
            root_embedding = get_face_recognition()(dict(user=Image.fromarray(np.uint8(main_image_copy))))[OutputKeys.IMG_EMBEDDING]
            compare_embedding = get_face_recognition()(dict(user=Image.fromarray(np.uint8(compare_image_copy))))[OutputKeys.IMG_EMBEDDING]
            score = float(np.dot(root_embedding, np.transpose(compare_embedding))[0][0])
        if result_prefix == "":
            result = str(round(score, 2))
        else:
            result = f"{result_prefix}_{round(score, 2)}"
        return (result,)
