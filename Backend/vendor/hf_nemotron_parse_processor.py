# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import numpy as np
from PIL import Image
from typing import List, Optional, Union, Dict, Any
import torch
from torchvision import transforms as T
import json

from transformers import BatchFeature, ProcessorMixin, BaseImageProcessor, ImageProcessingMixin
from transformers.image_utils import ChannelDimension, ImageInput, PILImageResampling, infer_channel_dimension_format
from transformers.utils import TensorType

OPENAI_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


from transformers.processing_utils import ImagesKwargs

class NemotronParseImagesKwargs(ImagesKwargs, total=False):
    content_hw: Optional[List[int]]

class NemotronParseImageProcessor(BaseImageProcessor, ImageProcessingMixin):
    """
    Image processor for NemotronParse model.
    
    This processor inherits from BaseImageProcessor to be compatible with transformers AutoImageProcessor.
    """
    
    valid_kwargs = NemotronParseImagesKwargs
    model_input_names = ["pixel_values", "image_sizes", "content_sizes"]
    
    def __init__(
        self,
        final_size: tuple = (2048, 1664),
        do_normalize: bool = False,
        image_mean: Optional[Union[List[float], tuple]] = None,
        image_std: Optional[Union[List[float], tuple]] = None,
        **kwargs,
    ):
        self.navit_packing = bool(kwargs.pop('navit_packing', False))
        self.navit_padding_mode = kwargs.pop('navit_padding_mode', 'top_left_white')
        self.navit_coordinate_frame = kwargs.pop('navit_coordinate_frame', 'content_normalized')
        clean_kwargs = {}
        for k, v in kwargs.items():
            if not k.startswith('_') and k not in ['transform', 'torch_transform']:
                clean_kwargs[k] = v
        
        if 'size' in clean_kwargs:
            size_config = clean_kwargs.pop('size')
            if isinstance(size_config, dict):
                if 'longest_edge' in size_config:
                    longest_edge = size_config['longest_edge']
                    if isinstance(longest_edge, (list, tuple)):
                        final_size = tuple(int(x) for x in longest_edge)
                    else:
                        final_size = (int(longest_edge), int(longest_edge))
                elif 'height' in size_config and 'width' in size_config:
                    final_size = (int(size_config['height']), int(size_config['width']))
        
        super().__init__(**clean_kwargs)
        self.do_normalize = bool(do_normalize)
        self.image_mean = list(image_mean or OPENAI_CLIP_MEAN)
        self.image_std = list(image_std or OPENAI_CLIP_STD)
        
        if isinstance(final_size, (list, tuple)) and len(final_size) >= 2:
            self.final_size = (int(final_size[0]), int(final_size[1]))
        elif isinstance(final_size, (int, float)):
            self.final_size = (int(final_size), int(final_size))
        else:
            self.final_size = (2048, 1664)  # Default fallback
        
        self._create_transforms()
    
    def _create_transforms(self):
        """Create transform objects (not serialized to JSON)."""
        if isinstance(self.final_size, (list, tuple)):
            self.target_height, self.target_width = int(self.final_size[0]), int(self.final_size[1])
        else:
            self.target_height = self.target_width = int(self.final_size)
        
        self.transform = None
        
        self.torch_transform = T.Compose([
            T.ToTensor(),
        ])

    def to_dict(self):
        """Override to exclude non-serializable transforms."""
        output = super().to_dict()
        output.pop('transform', None)
        output.pop('torch_transform', None)
        return output
    
    @classmethod
    def from_dict(cls, config_dict: dict, **kwargs):
        """Override to recreate transforms after loading."""
        config_dict = config_dict.copy()
        config_dict.pop('transform', None)
        config_dict.pop('torch_transform', None)
        
        # Clean any problematic entries
        for key in list(config_dict.keys()):
            if key.startswith('_') or config_dict[key] is None:
                config_dict.pop(key, None)
        
        # Ensure numeric types are correct
        if 'final_size' in config_dict:
            final_size = config_dict['final_size']
            if isinstance(final_size, (list, tuple)):
                config_dict['final_size'] = tuple(int(x) for x in final_size)
        if 'image_mean' in config_dict:
            config_dict['image_mean'] = [float(x) for x in config_dict['image_mean']]
        if 'image_std' in config_dict:
            config_dict['image_std'] = [float(x) for x in config_dict['image_std']]
        
        try:
            return cls(**config_dict, **kwargs)
        except Exception as e:
            print(f"Warning: Error in from_dict: {e}")
            print("Using default parameters...")
            return cls(**kwargs)
    
    def save_pretrained(self, save_directory, **kwargs):
        """Save image processor configuration."""
        import os
        import json
        
        os.makedirs(save_directory, exist_ok=True)
        
        # Save preprocessor config in standard HuggingFace format
        config = {
            "feature_extractor_type": "NemotronParseImageProcessor",
            "image_processor_type": "NemotronParseImageProcessor", 
            "processor_class": "NemotronParseProcessor",
            "size": {
                "height": self.final_size[0],
                "width": self.final_size[1],
                "longest_edge": self.final_size
            },
            "final_size": self.final_size,
            "do_normalize": self.do_normalize,
            "navit_packing": self.navit_packing,
            "navit_padding_mode": self.navit_padding_mode,
            "navit_coordinate_frame": self.navit_coordinate_frame,
            "image_mean": self.image_mean,
            "image_std": self.image_std,
        }
        
        config_path = os.path.join(save_directory, "preprocessor_config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

    def _resize_with_aspect_ratio(self, image: np.ndarray) -> np.ndarray:
        """Resize image maintaining aspect ratio (exact replica of original LongestMaxSizeHW)."""
        height, width = image.shape[:2]
        max_size_height = self.target_height
        max_size_width = self.target_width
        
        # Original LongestMaxSizeHW algorithm from custom_augmentations.py
        aspect_ratio = width / height
        new_height = height
        new_width = width

        if height > max_size_height:
            new_height = max_size_height
            new_width = int(new_height * aspect_ratio)

        if new_width > max_size_width:
            new_width = max_size_width
            new_height = int(new_width / aspect_ratio)
        
        if new_height == height and new_width == width:
            return image
        resampling = getattr(Image, "Resampling", Image).BILINEAR
        return np.asarray(Image.fromarray(image).resize((new_width, new_height), resampling))
    
    def _pad_to_size(self, image: np.ndarray) -> np.ndarray:
        """Center-pad image to target size with white padding."""
        h, w = image.shape[:2]
        min_height, min_width = self.target_height, self.target_width
        
        pad_h = max(0, min_height - h)
        pad_w = max(0, min_width - w)
        
        if pad_h == 0 and pad_w == 0:
            return image
        
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        
        if len(image.shape) == 3:
            return np.pad(
                image,
                ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                mode='constant',
                constant_values=255,
            )
        return np.pad(
            image,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode='constant',
            constant_values=255,
        )
    
    def _pad_to_navit_grid(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        target_height = ((height + 15) // 16) * 16
        target_width = ((width + 63) // 64) * 64
        pad_h, pad_w = target_height - height, target_width - width
        if not pad_h and not pad_w:
            return image
        padding = ((0, pad_h), (0, pad_w), (0, 0)) if image.ndim == 3 else ((0, pad_h), (0, pad_w))
        return np.pad(image, padding, mode="constant", constant_values=255)

    def preprocess(
        self,
        images: ImageInput,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Preprocess an image or batch of images for the NemotronParse model.
        
        Args:
            images: Input image(s)
            return_tensors: Type of tensors to return
        """
        
        captured_hw = kwargs.pop("content_hw", None)
        image_options = kwargs.pop("images_kwargs", {}) or {}
        nested_hw = image_options.get("content_hw")
        if captured_hw is not None and nested_hw is not None and list(captured_hw) != list(nested_hw):
            raise ValueError("conflicting captured content dimensions")
        if captured_hw is None:
            captured_hw = nested_hw
        if captured_hw is not None:
            if not self.navit_packing or self.navit_coordinate_frame != "content_normalized":
                raise ValueError("loader capture requires content-normalized NaViT")
            if self.navit_padding_mode != "top_left_white":
                raise ValueError("unsupported captured loader pixel contract")
            pages = images if isinstance(images, list) else [images]
            if len(pages) != 1 or len(captured_hw) != 2:
                raise ValueError("one captured page and [content_height, content_width] required")
            pixels = np.array(pages[0].convert("RGB"), copy=True)
            height, width = pixels.shape[:2]
            ch, cw = (int(value) for value in captured_hw)
            if list(captured_hw) != [ch, cw] or not (0 < ch <= height and 0 < cw <= width):
                raise ValueError("invalid captured content dimensions")
            if height % 16 or width % 64:
                raise ValueError("captured canvas is not aligned to the NaViT grid")
            page = torch.from_numpy(pixels).permute(2, 0, 1).contiguous().float().div_(255.0)
            # Native RADIO normalizes inside its encoder; the vLLM boundary
            # expects the equivalent normalization here, as in ordinary serving.
            if self.do_normalize:
                mean = page.new_tensor(self.image_mean).view(-1, 1, 1)
                std = page.new_tensor(self.image_std).view(-1, 1, 1)
                page = (page - mean) / std
            patches = page.unfold(1, 16, 16).unfold(2, 16, 16)
            patches = patches.permute(1, 2, 0, 3, 4).reshape(-1, 3 * 16 * 16)
            return BatchFeature(data={
                "pixel_values": patches,
                "image_sizes": torch.tensor([[height, width]], dtype=torch.long),
                "content_sizes": torch.tensor([[ch, cw]], dtype=torch.long),
            }, tensor_type=return_tensors)

        # Ensure images is a list
        if not isinstance(images, list):
            images = [images]

        # Ensure images are RGB
        for i in range(len(images)):
            images[i] = images[i].convert('RGB')
        
        # Convert PIL images to numpy arrays if needed
        processed_images = []
        for image in images:
            if isinstance(image, Image.Image):
                image = np.asarray(image)
            processed_images.append(image)
        
        # Apply NemotronParse-specific transforms
        pixel_values = []
        content_sizes = []
        for image in processed_images:
            processed_image = self._resize_with_aspect_ratio(image)
            content_sizes.append(
                [int(processed_image.shape[0]), int(processed_image.shape[1])]
            )
            if self.navit_packing:
                processed_image = self._pad_to_navit_grid(processed_image)
            
            if self.transform is not None:
                transformed = self.transform(image=processed_image)
                processed_image = transformed["image"]
            else:
                # Fixed exports retain the historical full-canvas pad.
                if not self.navit_packing:
                    processed_image = self._pad_to_size(processed_image)
            
            pixel_values_tensor = self.torch_transform(processed_image)
            
            if pixel_values_tensor.shape[0] == 1:
                pixel_values_tensor = pixel_values_tensor.expand(3, -1, -1)
                
            pixel_values.append(pixel_values_tensor)
                        
        if self.navit_packing:
            content_sizes = torch.tensor(content_sizes, dtype=torch.long)
            image_sizes = torch.tensor(
                [[int(page.shape[-2]), int(page.shape[-1])] for page in pixel_values],
                dtype=torch.long,
            )
            raw_patches = []
            for page in pixel_values:
                if self.do_normalize:
                    mean = page.new_tensor(self.image_mean).view(-1, 1, 1)
                    std = page.new_tensor(self.image_std).view(-1, 1, 1)
                    page = (page - mean) / std
                patches = page.unfold(1, 16, 16).unfold(2, 16, 16)
                patches = patches.permute(1, 2, 0, 3, 4).reshape(-1, 3 * 16 * 16)
                raw_patches.append(patches)
            pixel_values = torch.cat(raw_patches, dim=0)
            data = {
                "pixel_values": pixel_values,
                "image_sizes": image_sizes,
                "content_sizes": content_sizes,
            }
        else:
            pixel_values = torch.stack(pixel_values)
            if self.do_normalize:
                mean = pixel_values.new_tensor(self.image_mean).view(1, -1, 1, 1)
                std = pixel_values.new_tensor(self.image_std).view(1, -1, 1, 1)
                pixel_values = (pixel_values - mean) / std
            data = {"pixel_values": pixel_values}
        
        if return_tensors is not None:
            data = self._convert_output_format(data, return_tensors)
            
        return data
    
    def _convert_output_format(self, data: Dict[str, torch.Tensor], return_tensors: Union[str, TensorType]) -> Dict:
        """Convert output format based on return_tensors parameter."""
        if return_tensors == "pt" or return_tensors == TensorType.PYTORCH:
            return data
        elif return_tensors == "np" or return_tensors == TensorType.NUMPY:
            return {k: v.numpy() for k, v in data.items()}
        else:
            return data
    
    def __call__(self, images: Union[Image.Image, List[Image.Image]], **kwargs) -> Dict[str, torch.Tensor]:
        """Process images for the model (backward compatibility)."""
        return self.preprocess(images, **kwargs)


class NemotronParseProcessor(ProcessorMixin):
    
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = ("PreTrainedTokenizer", "PreTrainedTokenizerFast")
    
    def __init__(self, image_processor=None, tokenizer=None, **kwargs):
        if image_processor is None:
            image_processor = NemotronParseImageProcessor(**kwargs)
               
        super().__init__(image_processor, tokenizer)
            
    
    def __call__(
        self,
        images: Union[Image.Image, List[Image.Image]] = None,
        text: Union[str, List[str]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str] = False,
        truncation: Union[bool, str] = False,
        max_length: Optional[int] = None,
        stride: int = 0,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_token_type_ids: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        return_tensors: Optional[Union[str, "TensorType"]] = None,
        content_hw: Optional[List[int]] = None,
        **kwargs
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several text(s) and image(s).
        """
        
        # Process images
        if images is not None:
            image_inputs = self.image_processor(
                images, content_hw=content_hw, **kwargs)
        else:
            image_inputs = {}
        
        # Process text
        if text is not None:
            text_inputs = self.tokenizer(
                text,
                add_special_tokens=add_special_tokens,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                stride=stride,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
                return_offsets_mapping=return_offsets_mapping,
                return_token_type_ids=return_token_type_ids,
                return_length=return_length,
                verbose=verbose,
                return_tensors=return_tensors,
                **kwargs,
            )
        else:
            text_inputs = {}
        
        # Combine inputs
        return BatchFeature(data={**image_inputs, **text_inputs})
    
    def decode(self, *args, **kwargs):
        """Decode token ids to strings."""
        return self.tokenizer.decode(*args, **kwargs)
    
    def batch_decode(self, *args, **kwargs):
        """Batch decode token ids to strings."""
        return self.tokenizer.batch_decode(*args, **kwargs)
    
    def post_process_generation(self, sequences, fix_markdown=False):
        """Post-process generated sequences."""
        if hasattr(self.tokenizer, 'post_process_generation'):
            return self.tokenizer.post_process_generation(sequences, fix_markdown=fix_markdown)
        else:
            # Fallback processing
            if isinstance(sequences, str):
                sequences = [sequences]
            
            processed = []
            for seq in sequences:
                # Basic cleaning
                seq = seq.replace('<s>', '').replace('</s>', '').strip()
                processed.append(seq)
            
            return processed[0] if len(processed) == 1 else processed
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """
        Load processor from pretrained model.
        
        This method is compatible with AutoProcessor.from_pretrained().
        """
        # Explicitly load subcomponents via Auto* to ensure remote auto_map is honored.
        from transformers import AutoImageProcessor, AutoTokenizer
        trust_remote_code = kwargs.get("trust_remote_code", None)
        revision = kwargs.get("revision", None)
        token = kwargs.get("token", None)
        image_processor = AutoImageProcessor.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code,
            revision=revision,
            token=token,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code,
            revision=revision,
            token=token,
        )
        return cls(image_processor=image_processor, tokenizer=tokenizer)
    
    def save_pretrained(self, save_directory, **kwargs):
        """
        Save processor to directory.
        
        This method is compatible with AutoProcessor/AutoImageProcessor loading.
        """
        import os
        os.makedirs(save_directory, exist_ok=True)
        
        # Save tokenizer with proper configuration for AutoTokenizer
        print("Saving tokenizer for AutoTokenizer compatibility...")
        self.tokenizer.save_pretrained(save_directory, **kwargs)
        
        # Save image processor  
        print("Saving image processor...")
        self.image_processor.save_pretrained(save_directory, **kwargs)
        
        # Use the parent class's save_pretrained method for processor config
        super().save_pretrained(save_directory, **kwargs)
        print(f"NemotronParseProcessor saved to {save_directory}")
        print(f"AutoTokenizer.from_pretrained('{save_directory}') should now work!")
