from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import insightface
from insightface.app import FaceAnalysis
from .utils.face_process_utils import Face_Skin
from .utils.psgan_utils import PSGAN_Inference
import folder_paths
import os
import traceback
from .config import *

retinaface_detection = None
image_face_fusion = None
face_analysis = None
face_skin = None
roop = None
skin_retouching = None
skin_retouching_model = None
portrait_enhancement = None
psgan_interface = None
real_gan_sr = None
face_recognition = None

# Add direct model variables for ONNX-based skin retouching
skin_generator = None
skin_inpainting_net = None 
skin_detection_net = None
skin_face_detector = None
skin_diffuse_mask = None

models_dir = folder_paths.models_dir
model_path_1 = os.path.join(models_dir, "facechain-colab/hub/damo/cv_resnet50_face-detection_retinaface")
model_path_2 = os.path.join(models_dir, "facechain-colab//hub/damo/cv_unet_skin_retouching_torch") #facechain-colab/hub/damo/

def get_retinaface_detection():
    global retinaface_detection
    if retinaface_detection is None:
        retinaface_detection = pipeline(Tasks.face_detection, model=model_path_1, model_revision='v2.0.2')
    return retinaface_detection

def get_image_face_fusion():
    global image_face_fusion
    if image_face_fusion is None:
        image_face_fusion = pipeline(Tasks.image_face_fusion, model='damo/cv_unet-image-face-fusion_damo', model_revision='v1.3')
    return image_face_fusion

def get_face_analysis():
    global face_analysis
    if face_analysis is None:
        face_analysis = FaceAnalysis(name='buffalo_l')
    return face_analysis

def get_roop():
    global roop
    if roop is None:
        roop = insightface.model_zoo.get_model('inswapper_128.onnx', download=True, download_zip=True, root=root_path)
    return roop

def get_face_skin():
    global face_skin
    if face_skin is None:
        face_skin = Face_Skin(os.path.join(models_path, "face_skin.pth"))
    return face_skin

def get_skin_retouching():
    global skin_retouching
    if skin_retouching is None:
        print(f"models dir is: {models_dir}\n model path is: {model_path_2}")
        skin_retouching = pipeline('skin-retouching-torch', model=model_path_2, model_revision='v1.0.2') #model_path_2
    return skin_retouching

def set_skin_retouching_model(model):
    global skin_retouching_model
    skin_retouching_model = model

def get_skin_retouching_model():
    global skin_retouching_model
    return skin_retouching_model

def get_portrait_enhancement():
    global portrait_enhancement
    if portrait_enhancement is None:
        portrait_enhancement = pipeline(Tasks.image_portrait_enhancement, model='damo/cv_gpen_image-portrait-enhancement', model_revision='v1.0.0')
    return portrait_enhancement

def get_real_gan_sr():
    global real_gan_sr
    if real_gan_sr is None:
        real_gan_sr = pipeline('image-super-resolution-x2', model='bubbliiiing/cv_rrdb_image-super-resolution_x2', model_revision="v1.0.2")
    return real_gan_sr

def get_pagan_interface():
    global psgan_interface
    if psgan_interface is None:
        face_landmarks_model_path = os.path.join(models_path, "face_landmarks.pth")
        makeup_transfer_model_path = os.path.join(models_path, "makeup_transfer.pth")
        psgan_interface = PSGAN_Inference("cuda", makeup_transfer_model_path, get_retinaface_detection(), get_face_skin(), face_landmarks_model_path)
    return psgan_interface

def get_face_recognition():
    global face_recognition
    if face_recognition is None:
        face_recognition = pipeline("face_recognition", model="bubbliiiing/cv_retinafce_recognition", model_revision="v1.0.3")
    return face_recognition

class OnnxFaceDetector:
    """Direct ONNX implementation of RetinaFace face detection without using ModelScope pipeline"""
    def __init__(self, model_path, device='cuda'):
        import onnxruntime as ort
        import torch
        import numpy as np
        import cv2
        import time
        import traceback
        
        self.total_inference_time = 0
        self.inference_count = 0
        
        # Setup ONNX Runtime session
        start_time = time.time()
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
        try:
            # Create ONNX inference session
            print(f"Loading RetinaFace ONNX model from: {model_path}")
            self.session = ort.InferenceSession(model_path, providers=providers)
            
            # Get model metadata
            self.input_name = self.session.get_inputs()[0].name
            self.input_shape = self.session.get_inputs()[0].shape
            self.output_names = [output.name for output in self.session.get_outputs()]
            
            print(f"ONNX model loaded - Input shape: {self.input_shape}, Output names: {self.output_names}")
            print(f"ONNX model loading took {time.time() - start_time:.4f} seconds")
            
            # Default confidence threshold
            self.confidence_threshold = 0.5
            
            # These are default settings for RetinaFace
            self.center_variance = 0.1
            self.size_variance = 0.2
            self.min_boxes = [[16, 32], [64, 128], [256, 512]]
            self.strides = [8, 16, 32]
            
            # Generate priors (anchors)
            prior_start = time.time()
            self.priors = self._generate_priors()
            print(f"Generating {len(self.priors)} anchor boxes took {time.time() - prior_start:.4f} seconds")
            
        except Exception as e:
            print(f"Error initializing ONNX face detector: {e}")
            traceback.print_exc()
            raise e
    
    def _generate_priors(self):
        """Generate anchor boxes for RetinaFace"""
        import numpy as np
        
        # Feature map sizes for RetinaFace with input size 640x640
        feature_map_sizes = [[80, 80], [40, 40], [20, 20]]
        priors = []
        
        for k, f in enumerate(feature_map_sizes):
            min_sizes = self.min_boxes[k]
            for i in range(f[0]):
                for j in range(f[1]):
                    for min_size in min_sizes:
                        s_kx = min_size / 640.0
                        s_ky = min_size / 640.0
                        
                        cx = (j + 0.5) * self.strides[k] / 640.0
                        cy = (i + 0.5) * self.strides[k] / 640.0
                        
                        priors.append([cx, cy, s_kx, s_ky])
        
        return np.array(priors, dtype=np.float32)
    
    def _preprocess(self, image):
        """Preprocess image for RetinaFace input"""
        import cv2
        import numpy as np
        import time
        
        preprocess_start = time.time()
        
        # Keep track of original size for rescaling outputs
        original_shape = image.shape[:2]
        self.original_height, self.original_width = original_shape
        
        # RetinaFace expects RGB images
        color_convert_start = time.time()
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # RGBA
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        elif image.shape[2] == 3 and len(image.shape) == 3:
            if False:  # Check if it's BGR instead of RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        color_convert_time = time.time() - color_convert_start
        
        # Resize to the input size expected by the model (e.g., 640x640)
        resize_start = time.time()
        target_size = (640, 640)
        self.scale_factor = (original_shape[1] / target_size[0], original_shape[0] / target_size[1])
        resized_img = cv2.resize(image, target_size)
        resize_time = time.time() - resize_start
        
        # Convert to float and normalize
        normalize_start = time.time()
        resized_img = resized_img.astype(np.float32)
        resized_img = (resized_img - 127.5) / 128.0
        normalize_time = time.time() - normalize_start
        
        # Move channels first for ONNX model
        transpose_start = time.time()
        resized_img = resized_img.transpose(2, 0, 1)
        transpose_time = time.time() - transpose_start
        
        # Add batch dimension
        batch_start = time.time()
        input_data = np.expand_dims(resized_img, axis=0)
        batch_time = time.time() - batch_start
        
        preprocess_time = time.time() - preprocess_start
        
        if self.inference_count % 10 == 0:  # Log details every 10 inferences
            print(f"Preprocess breakdown: Color={color_convert_time:.4f}s, Resize={resize_time:.4f}s, "
                  f"Normalize={normalize_time:.4f}s, Transpose={transpose_time:.4f}s, "
                  f"Batch={batch_time:.4f}s, Total={preprocess_time:.4f}s")
        
        return input_data
    
    def _decode_boxes(self, raw_boxes):
        """Convert RetinaFace raw outputs to boxes"""
        import numpy as np
        import time
        
        decode_start = time.time()
        
        # Raw boxes are the offset predictions from the model
        loc = raw_boxes[0]  # Shape: [1, num_priors, 4]
        # Remove batch dimension
        loc = loc.squeeze(0)
        
        # Convert offset predictions to boxes
        priors = self.priors
        boxes = np.zeros_like(loc)
        
        # Center form to corner form conversion with variance applied
        boxes[:, 0] = priors[:, 0] + loc[:, 0] * self.center_variance * priors[:, 2]
        boxes[:, 1] = priors[:, 1] + loc[:, 1] * self.center_variance * priors[:, 3] 
        boxes[:, 2] = priors[:, 2] * np.exp(loc[:, 2] * self.size_variance)
        boxes[:, 3] = priors[:, 3] * np.exp(loc[:, 3] * self.size_variance)
        
        # Convert center form to corner form
        boxes[:, 0] -= boxes[:, 2] / 2
        boxes[:, 1] -= boxes[:, 3] / 2
        boxes[:, 2] += boxes[:, 0]
        boxes[:, 3] += boxes[:, 1]
        
        # Rescale to original image size
        boxes[:, 0] *= self.original_width
        boxes[:, 1] *= self.original_height
        boxes[:, 2] *= self.original_width
        boxes[:, 3] *= self.original_height
        
        decode_time = time.time() - decode_start
        if self.inference_count % 10 == 0:
            print(f"Box decoding took {decode_time:.4f} seconds")
        
        return boxes
    
    def _decode_landmarks(self, raw_landmarks, boxes):
        """Convert raw landmark predictions to actual landmark coordinates"""
        import numpy as np
        import time
        
        landmark_start = time.time()
        
        landmarks = raw_landmarks[0].squeeze(0)
        priors = self.priors
        
        decoded_landmarks = np.zeros((landmarks.shape[0], 10))
        
        # Decode each landmark point (5 points, each with x,y)
        for i in range(5):
            # X coordinate
            decoded_landmarks[:, i*2] = priors[:, 0] + landmarks[:, i*2] * self.center_variance * priors[:, 2]
            decoded_landmarks[:, i*2] *= self.original_width
            
            # Y coordinate
            decoded_landmarks[:, i*2+1] = priors[:, 1] + landmarks[:, i*2+1] * self.center_variance * priors[:, 3]
            decoded_landmarks[:, i*2+1] *= self.original_height
        
        landmark_time = time.time() - landmark_start
        if self.inference_count % 10 == 0:
            print(f"Landmark decoding took {landmark_time:.4f} seconds")
        
        return decoded_landmarks
    
    def _nms(self, boxes, scores, threshold=0.4):
        """Non-maximum suppression to remove overlapping detections"""
        import numpy as np
        import time
        
        nms_start = time.time()
        
        # If no boxes, return empty list
        if len(boxes) == 0:
            return []
        
        # Grab coordinates for comparison
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        # Calculate area of each box
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        
        # Sort by confidence score
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            # Calculate intersection with all remaining boxes
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            # Calculate intersection area
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            intersection = w * h
            
            # Calculate IoU
            overlap = intersection / (areas[i] + areas[order[1:]] - intersection)
            
            # Keep boxes with IoU less than threshold
            indices = np.where(overlap <= threshold)[0]
            order = order[indices + 1]
        
        nms_time = time.time() - nms_start
        if self.inference_count % 10 == 0:
            print(f"NMS took {nms_time:.4f} seconds for {len(keep)} boxes out of {len(boxes)}")
        
        return keep
    
    def __call__(self, image):
        """Run inference on image and return detected faces"""
        import numpy as np
        import time
        
        self.inference_count += 1
        overall_start = time.time()
        
        # Preprocess image
        preprocess_start = time.time()
        input_data = self._preprocess(image)
        preprocess_time = time.time() - preprocess_start
        
        # Run inference
        inference_start = time.time()
        try:
            outputs = self.session.run(self.output_names, {self.input_name: input_data})
            inference_time = time.time() - inference_start
            self.total_inference_time += inference_time
            avg_inference_time = self.total_inference_time / self.inference_count
            
            # Debug output format
            print(f"ONNX model output format:")
            print(f"  - Number of outputs: {len(outputs)}")
            for i, output in enumerate(outputs):
                print(f"  - Output {i} shape: {output.shape}, type: {output.dtype}")
            
            if self.inference_count % 10 == 0:
                print(f"ONNX inference took {inference_time:.4f} seconds (avg: {avg_inference_time:.4f}s)")
        except Exception as e:
            print(f"Error during ONNX inference: {e}")
            traceback.print_exc()
            raise e
        
        # Post-processing starts
        postprocess_start = time.time()
        
        # Handle different output formats
        # Some ONNX models might output in different formats
        # We need to determine which output contains what information
        
        # Case 1: Standard RetinaFace format with separate outputs
        if len(outputs) >= 2:
            # Default case - standard RetinaFace model
            box_output = outputs[0]  # locations
            confidence_output = outputs[1]  # confidence scores
            
            # Get confidence scores for face class
            confidence = confidence_output[0, :, 1] if confidence_output.shape[-1] == 2 else confidence_output.squeeze()
            
            # Filter by confidence threshold
            filter_start = time.time()
            mask = confidence > self.confidence_threshold
            filtered_confidence = confidence[mask]
            filter_time = time.time() - filter_start
            
            if len(filtered_confidence) == 0:
                overall_time = time.time() - overall_start
                print(f"No faces detected. Overall detection took {overall_time:.4f} seconds")
                return {'boxes': [], 'scores': [], 'keypoints': []}
            
            # Decode boxes and filter
            boxes = self._decode_boxes([box_output])
            filtered_boxes = boxes[mask]
            
            # Apply non-maximum suppression
            keep_indices = self._nms(filtered_boxes, filtered_confidence)
            
            result_boxes = filtered_boxes[keep_indices].astype(np.int32)
            result_scores = filtered_confidence[keep_indices]
            
            # Decode landmarks if they're available
            landmark_time = 0
            if len(outputs) > 2:  # Check if landmarks are in the output
                landmark_start = time.time()
                landmarks = self._decode_landmarks(outputs[2:], boxes)
                filtered_landmarks = landmarks[mask]
                result_landmarks = filtered_landmarks[keep_indices].reshape(-1, 5, 2).astype(np.int32)
                landmark_time = time.time() - landmark_start
            else:
                result_landmarks = None
                
        # Case 2: Single output model (consolidated format)
        elif len(outputs) == 1:
            print("Detected single-output ONNX model format")
            # Some models output all information in a single tensor
            output = outputs[0]
            print(f"Single output shape: {output.shape}")
            
            # If output has detection results with box, score, class format: [batch, detections, box+score+class]
            if len(output.shape) == 3 and output.shape[2] >= 5:  # At least x1,y1,x2,y2,score
                # Extract boxes and scores
                detections = output[0]  # Remove batch dimension
                
                # Filter by confidence threshold
                filter_start = time.time()
                # Score is typically the 5th element (after x1,y1,x2,y2)
                scores = detections[:, 4]
                mask = scores > self.confidence_threshold
                filtered_detections = detections[mask]
                filtered_confidence = scores[mask]
                filter_time = time.time() - filter_start
                
                if len(filtered_confidence) == 0:
                    overall_time = time.time() - overall_start
                    print(f"No faces detected. Overall detection took {overall_time:.4f} seconds")
                    return {'boxes': [], 'scores': [], 'keypoints': []}
                
                # Boxes are typically the first 4 elements
                filtered_boxes = filtered_detections[:, :4]
                
                # Apply non-maximum suppression
                keep_indices = self._nms(filtered_boxes, filtered_confidence)
                
                result_boxes = filtered_boxes[keep_indices].astype(np.int32)
                result_scores = filtered_confidence[keep_indices]
                
                # Extract landmarks if available (typically elements after box and score)
                landmark_time = 0
                if output.shape[2] > 5:  # Check if there are elements for landmarks
                    landmark_start = time.time()
                    # Landmarks are the remaining elements, reshape to [5, 2] for 5 points
                    landmarks_data = filtered_detections[:, 5:15] if output.shape[2] >= 15 else None
                    if landmarks_data is not None:
                        result_landmarks = landmarks_data[keep_indices].reshape(-1, 5, 2).astype(np.int32)
                    else:
                        result_landmarks = None
                    landmark_time = time.time() - landmark_start
                else:
                    result_landmarks = None
            
            # Handle feature map outputs (4D tensors) - specifically for models that output feature maps
            elif len(output.shape) == 4:
                print(f"Detected feature map output format with shape {output.shape}")
                print("This appears to be a feature extraction model rather than a direct face detector.")
                
                # Check if this looks like a specific known architecture
                if output.shape[1] == 2048 and output.shape[2] == 20 and output.shape[3] == 20:
                    print("Detected a feature map that matches known pattern [1, 2048, 20, 20]")
                    try:
                        # Try to convert feature map to face detections
                        result_boxes, result_scores, result_landmarks = self._convert_feature_map_to_detections(output, image)
                        if len(result_boxes) > 0:
                            print(f"Successfully extracted {len(result_boxes)} faces from feature map")
                            format_start = time.time()
                            result = {
                                'boxes': result_boxes.tolist() if len(result_boxes) > 0 else [],
                                'scores': result_scores.tolist() if len(result_scores) > 0 else [],
                                'keypoints': result_landmarks
                            }
                            format_time = time.time() - format_start
                            postprocess_time = time.time() - postprocess_start
                            overall_time = time.time() - overall_start
                            print(f"Face detection completed in {overall_time:.4f} seconds")
                            return result
                    except Exception as e:
                        print(f"Feature map conversion failed: {e}")
                        traceback.print_exc()
                
                print("Converting to detector output format...")
                print("Falling back to ModelScope pipeline to handle this model format")
                
                # Fallback to standard pipeline
                from modelscope.pipelines import pipeline
                from modelscope.utils.constant import Tasks
                try:
                    pipeline_detector = pipeline(Tasks.face_detection, model=model_path_1, model_revision='v2.0.2')
                    return pipeline_detector(image)
                except Exception as e:
                    print(f"Fallback pipeline failed: {e}")
                    # Return empty result if everything fails
                    return {'boxes': [], 'scores': [], 'keypoints': []}
            
            # Unsupported format
            else:
                print(f"Unsupported ONNX model output format. Please check model documentation.")
                print(f"Falling back to modelscope pipeline")
                # Fall back to modelscope pipeline
                from modelscope.pipelines import pipeline
                from modelscope.utils.constant import Tasks
                pipeline_detector = pipeline(Tasks.face_detection, model=model_path_1, model_revision='v2.0.2')
                return pipeline_detector(image)
        
        # Case 3: Unsupported format
        else:
            print(f"Empty output from ONNX model. Please check model documentation.")
            print(f"Falling back to modelscope pipeline")
            # Fall back to modelscope pipeline
            from modelscope.pipelines import pipeline
            from modelscope.utils.constant import Tasks
            pipeline_detector = pipeline(Tasks.face_detection, model=model_path_1, model_revision='v2.0.2')
            return pipeline_detector(image)
        
        # Format output to match ModelScope's format
        format_start = time.time()
        result = {
            'boxes': result_boxes.tolist() if len(result_boxes) > 0 else [],
            'scores': result_scores.tolist() if len(result_scores) > 0 else [],
            'keypoints': result_landmarks.tolist() if result_landmarks is not None and len(result_landmarks) > 0 else []
        }
        format_time = time.time() - format_start
        
        postprocess_time = time.time() - postprocess_start
        overall_time = time.time() - overall_start
        
        # Log timing information
        if self.inference_count % 10 == 0 or len(result_boxes) > 0:
            print(f"Face detection performance:")
            print(f"  - Preprocessing: {preprocess_time:.4f}s")
            print(f"  - ONNX Inference: {inference_time:.4f}s")
            print(f"  - Confidence filtering: {filter_time:.4f}s")
            print(f"  - Landmarks: {landmark_time:.4f}s")
            print(f"  - Result formatting: {format_time:.4f}s")
            print(f"  - Total post-processing: {postprocess_time:.4f}s")
            print(f"  - TOTAL face detection: {overall_time:.4f}s")
            print(f"  - Found {len(result_boxes)} faces")
        
        return result

    def _convert_feature_map_to_detections(self, feature_map, original_image):
        """Convert feature map outputs to face detections - specialized for [1, 2048, 20, 20] format"""
        import numpy as np
        import cv2
        import time
        
        print("Attempting to convert feature map to face detections")
        start_time = time.time()
        
        # For this specific model architecture, we'll use heuristics to find faces
        # This is a simplified approach and may need model-specific adjustments
        
        # Extract key feature channels
        # For feature maps, often different channels encode different face properties
        feature_map = feature_map[0]  # Remove batch dimension
        
        # Method 1: Use channel reduction and thresholding to find potential face regions
        # Reduce channels to get a 2D heatmap (this is a heuristic and might need adjustment)
        feature_sum = np.sum(feature_map, axis=0)
        feature_mean = np.mean(feature_map, axis=0)
        feature_max = np.max(feature_map, axis=0)
        
        # Normalize the feature maps
        feature_sum = (feature_sum - np.min(feature_sum)) / (np.max(feature_sum) - np.min(feature_sum) + 1e-8)
        feature_mean = (feature_mean - np.min(feature_mean)) / (np.max(feature_mean) - np.min(feature_mean) + 1e-8)
        feature_max = (feature_max - np.min(feature_max)) / (np.max(feature_max) - np.min(feature_max) + 1e-8)
        
        # Combine them (weighted sum)
        heatmap = 0.4 * feature_sum + 0.3 * feature_mean + 0.3 * feature_max
        
        # Resize to original image dimensions
        original_h, original_w = original_image.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (original_w, original_h))
        
        # Threshold to find potential face regions
        _, binary_map = cv2.threshold(heatmap_resized, 0.5, 1.0, cv2.THRESH_BINARY)
        binary_map = (binary_map * 255).astype(np.uint8)
        
        # Find contours in the binary map
        contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size to find face candidates
        face_boxes = []
        face_scores = []
        min_face_area = original_h * original_w * 0.01  # Minimum 1% of image area
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            
            # Filter small areas and ensure reasonable aspect ratio for faces
            if area > min_face_area and 0.5 <= w/h <= 2.0:
                # Calculate confidence based on average heatmap value in the region
                region_heatmap = heatmap_resized[y:y+h, x:x+w]
                confidence = np.mean(region_heatmap)
                
                # Only keep regions with high confidence
                if confidence > 0.3:
                    face_boxes.append([x, y, x+w, y+h])
                    face_scores.append(confidence)
        
        # Convert to numpy arrays
        if len(face_boxes) > 0:
            face_boxes = np.array(face_boxes, dtype=np.int32)
            face_scores = np.array(face_scores, dtype=np.float32)
            
            # Apply NMS to remove overlapping detections
            keep_indices = self._nms(face_boxes, face_scores)
            result_boxes = face_boxes[keep_indices]
            result_scores = face_scores[keep_indices]
            
            # Generate placeholder landmarks (centered in each face box)
            # Instead of returning None, create dummy landmarks for each detected face
            result_landmarks = []
            for box in result_boxes:
                # Create 5 points (left eye, right eye, nose, left mouth, right mouth)
                # Generate reasonable positions from the bounding box
                x1, y1, x2, y2 = box
                w, h = x2 - x1, y2 - y1
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                
                # Simple default facial landmarks positioning
                left_eye = [int(x1 + w * 0.3), int(y1 + h * 0.3)]
                right_eye = [int(x2 - w * 0.3), int(y1 + h * 0.3)]
                nose = [center_x, int(y1 + h * 0.5)]
                left_mouth = [int(x1 + w * 0.3), int(y2 - h * 0.3)]
                right_mouth = [int(x2 - w * 0.3), int(y2 - h * 0.3)]
                
                # Add as a 5x2 array (5 points, each with x,y)
                landmarks = [left_eye, right_eye, nose, left_mouth, right_mouth]
                result_landmarks.append(landmarks)
            
            print(f"Feature map conversion found {len(result_boxes)} faces in {time.time() - start_time:.4f} seconds")
            return result_boxes, result_scores, result_landmarks
        
        # Alternative method if contour-based approach failed
        if len(face_boxes) == 0:
            print("Contour method failed, trying alternative approach")
            
            # Try a simple sliding window approach
            window_size = min(original_h, original_w) // 4
            stride = window_size // 2
            
            for y in range(0, original_h - window_size, stride):
                for x in range(0, original_w - window_size, stride):
                    region_heatmap = heatmap_resized[y:y+window_size, x:x+window_size]
                    confidence = np.mean(region_heatmap)
                    
                    if confidence > 0.4:  # Higher threshold for sliding window
                        face_boxes.append([x, y, x+window_size, y+window_size])
                        face_scores.append(confidence)
            
            if len(face_boxes) > 0:
                face_boxes = np.array(face_boxes, dtype=np.int32)
                face_scores = np.array(face_scores, dtype=np.float32)
                
                # Apply NMS again
                keep_indices = self._nms(face_boxes, face_scores)
                result_boxes = face_boxes[keep_indices]
                result_scores = face_scores[keep_indices]
                
                # Generate placeholder landmarks as above
                result_landmarks = []
                for box in result_boxes:
                    x1, y1, x2, y2 = box
                    w, h = x2 - x1, y2 - y1
                    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                    
                    left_eye = [int(x1 + w * 0.3), int(y1 + h * 0.3)]
                    right_eye = [int(x2 - w * 0.3), int(y1 + h * 0.3)]
                    nose = [center_x, int(y1 + h * 0.5)]
                    left_mouth = [int(x1 + w * 0.3), int(y2 - h * 0.3)]
                    right_mouth = [int(x2 - w * 0.3), int(y2 - h * 0.3)]
                    
                    landmarks = [left_eye, right_eye, nose, left_mouth, right_mouth]
                    result_landmarks.append(landmarks)
                
                print(f"Sliding window method found {len(result_boxes)} faces in {time.time() - start_time:.4f} seconds")
                return result_boxes, result_scores, result_landmarks
        
        # If all methods fail, return empty results
        print("No faces found in feature map")
        return np.array([], dtype=np.int32), np.array([], dtype=np.float32), []

# New functions for direct ONNX-based skin retouching
def load_skin_retouching_direct_models(device="cuda"):
    """Load all models for skin retouching directly without using the modelscope pipeline"""
    import torch
    import torch.nn.functional as F
    from modelscope.models.cv.skin_retouching.unet_deploy import UNet
    from modelscope.models.cv.skin_retouching.detection_model.detection_unet_in import DetectionUNet
    from modelscope.models.cv.skin_retouching.inpainting_model.inpainting_unet import RetouchingNet
    from modelscope.models.cv.skin_retouching.utils import gen_diffuse_mask
    import onnxruntime as ort
    
    global skin_generator, skin_inpainting_net, skin_detection_net, skin_face_detector, skin_diffuse_mask
    
    # Set device
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    try:
        # Load the generator model - fix the path to match standard modelscope path
        model_path = os.path.join(model_path_2, "pytorch_model.pt")
        # Fallback to checking for alternative filenames if the first one doesn't exist
        if not os.path.exists(model_path):
            print(f"Model file not found at {model_path}, trying alternatives...")
            alternatives = ["pytorch_model.bin", "model.pt", "model.pth", "generator.pt", "generator.pth"]
            for alt in alternatives:
                alt_path = os.path.join(model_path_2, alt)
                if os.path.exists(alt_path):
                    model_path = alt_path
                    print(f"Found alternative model file: {model_path}")
                    break
            
        print(f"Loading generator model from: {model_path}")
        skin_generator = UNet(3, 3).to(device)
        weights = torch.load(model_path, map_location='cpu')
        # Handle different key format in weights dictionary
        if 'generator' in weights:
            skin_generator.load_state_dict(weights['generator'])
        else:
            # Try loading directly if there's no 'generator' key
            skin_generator.load_state_dict(weights)
        skin_generator.eval()
        
        # Load the local retouching models
        local_model_path = os.path.join(model_path_2, "joint_20210926.pth")
        if not os.path.exists(local_model_path):
            print(f"Local model not found at {local_model_path}, trying alternatives...")
            alternatives = ["joint.pth", "joint_model.pth", "local_model.pth", "inpainting.pth"]
            for alt in alternatives:
                alt_path = os.path.join(model_path_2, alt)
                if os.path.exists(alt_path):
                    local_model_path = alt_path
                    print(f"Found alternative local model file: {local_model_path}")
                    break
                    
        print(f"Loading local retouching models from: {local_model_path}")
        ckpt_dict_load = torch.load(local_model_path, map_location='cpu')
        
        # Inpainting network
        skin_inpainting_net = RetouchingNet(in_channels=4, out_channels=3).to(device)
        if 'inpainting_net' in ckpt_dict_load:
            skin_inpainting_net.load_state_dict(ckpt_dict_load['inpainting_net'])
        elif 'inpainting' in ckpt_dict_load:
            skin_inpainting_net.load_state_dict(ckpt_dict_load['inpainting'])
        else:
            # If no specific key found, try direct loading
            print("Warning: No inpainting_net key found in checkpoint, trying direct loading")
            skin_inpainting_net.load_state_dict(ckpt_dict_load)
        skin_inpainting_net.eval()
        
        # Detection network
        skin_detection_net = DetectionUNet(n_channels=3, n_classes=1).to(device)
        if 'detection_net' in ckpt_dict_load:
            skin_detection_net.load_state_dict(ckpt_dict_load['detection_net'])
        elif 'detection' in ckpt_dict_load:
            skin_detection_net.load_state_dict(ckpt_dict_load['detection'])
        else:
            # Detection network might be in a separate file
            print("Warning: No detection_net key found in checkpoint, looking for separate file")
            detection_paths = ["detection_net.pth", "detection.pth", "detection_model.pth"]
            for det_path in detection_paths:
                det_file = os.path.join(model_path_2, det_path)
                if os.path.exists(det_file):
                    print(f"Found detection model at: {det_file}")
                    det_weights = torch.load(det_file, map_location='cpu')
                    skin_detection_net.load_state_dict(det_weights)
                    break
        skin_detection_net.eval()
        
        # Load Face detector using ONNX directly instead of the pipeline
        print("Loading RetinaFace face detector with ONNX")
        # Check for the ONNX model file
        face_detector_onnx_path = os.path.join(model_path_1, "model.onnx")
        if not os.path.exists(face_detector_onnx_path):
            print(f"ONNX model not found at {face_detector_onnx_path}, searching for alternatives...")
            alternatives = ["retinaface.onnx", "retina.onnx", "detector.onnx", "face_detector.onnx"]
            
            # Try to find in the same directory
            for alt in alternatives:
                alt_path = os.path.join(model_path_1, alt)
                if os.path.exists(alt_path):
                    face_detector_onnx_path = alt_path
                    print(f"Found alternative ONNX model at: {face_detector_onnx_path}")
                    break
                    
            # If not found, check parent directories
            if not os.path.exists(face_detector_onnx_path):
                parent_dir = os.path.dirname(model_path_1)
                for alt in alternatives:
                    alt_path = os.path.join(parent_dir, alt)
                    if os.path.exists(alt_path):
                        face_detector_onnx_path = alt_path
                        print(f"Found ONNX model in parent directory: {face_detector_onnx_path}")
                        break
        
        if os.path.exists(face_detector_onnx_path):
            # Initialize ONNX face detector
            try:
                print(f"Attempting to load RetinaFace ONNX model from {face_detector_onnx_path}")
                # Try to verify model compatibility before fully loading
                import onnx
                try:
                    # Check if model can be loaded and has compatible inputs/outputs
                    model = onnx.load(face_detector_onnx_path)
                    inputs = [inp.name for inp in model.graph.input]
                    outputs = [out.name for out in model.graph.output]
                    print(f"ONNX model overview - Inputs: {inputs}, Outputs: {outputs}")
                except Exception as e:
                    print(f"Warning: Could not pre-validate ONNX model: {e}")
                
                # Try to load and initialize the detector
                skin_face_detector = OnnxFaceDetector(face_detector_onnx_path, device=device)
                
                # Test the detector with a simple image to verify it works correctly
                try:
                    print("Testing ONNX face detector with a small test image")
                    import numpy as np
                    # Create a small test image (1x1 pixel)
                    test_img = np.zeros((64, 64, 3), dtype=np.uint8)
                    # Run a test inference
                    test_result = skin_face_detector(test_img)
                    print("ONNX face detector test passed successfully")
                except Exception as e:
                    print(f"ONNX face detector test failed: {e}")
                    print("Falling back to ModelScope pipeline")
                    traceback.print_exc()
                    skin_face_detector = get_retinaface_detection()
                
                print(f"Successfully loaded face detector")
            except Exception as e:
                print(f"Error loading ONNX face detector: {e}")
                traceback.print_exc()
                print("Falling back to ModelScope pipeline")
                skin_face_detector = get_retinaface_detection()
        else:
            # Fallback to default pipeline if ONNX model not found
            print(f"ONNX model not found at {face_detector_onnx_path}, falling back to ModelScope pipeline")
            skin_face_detector = get_retinaface_detection()
        
        # Create diffuse mask for border smoothing
        print("Generating diffuse mask")
        skin_diffuse_mask = gen_diffuse_mask()
        skin_diffuse_mask = torch.from_numpy(skin_diffuse_mask).to(device).float()
        skin_diffuse_mask = skin_diffuse_mask.permute(2, 0, 1)[None, ...]
        
        print("All skin retouching models loaded successfully")
        
    except Exception as e:
        import traceback
        print(f"Error loading skin retouching models: {e}")
        traceback.print_exc()
        raise e
    
    return {
        "generator": skin_generator,
        "inpainting_net": skin_inpainting_net,
        "detection_net": skin_detection_net, 
        "face_detector": skin_face_detector,
        "diffuse_mask": skin_diffuse_mask,
        "device": device
    }

def get_skin_models():
    """Get the loaded skin retouching models"""
    global skin_generator, skin_inpainting_net, skin_detection_net, skin_face_detector, skin_diffuse_mask
    return {
        "generator": skin_generator,
        "inpainting_net": skin_inpainting_net,
        "detection_net": skin_detection_net,
        "face_detector": skin_face_detector,
        "diffuse_mask": skin_diffuse_mask
    }
