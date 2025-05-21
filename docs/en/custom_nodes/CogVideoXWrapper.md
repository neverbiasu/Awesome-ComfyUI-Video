# 📹 ComfyUI-VideoHelperSuite

## 🔍 Suite Overview

| Item       | Details                                                                  |
| ---------- | --------------------------------------------------------------------- |
| 📌 **Author** | [@Kkijai](https://github.com/kijai)                        |
| 📅 **Version** | 1.2.0+                                                                |
| 🏷️ **Category** | Open Source Model Extension                                              |
| 🔗 **Repository** | [GitHub Link](https://github.com/kijai/ComfyUI-CogVideoXWrapper) |

## 📝 Functional Introduction
An open-source extension developed by kijai, it seamlessly integrates the cutting-edge CogVideoX large-scale text-to-video model into the node-based ComfyUI interface. It features a complete set of custom nodes for model downloading, input encoding, sampling inference, and result decoding, enabling diverse functionalities such as "text-to-video generation," "image-to-video generation," and "video style transfer."

## 📊 Node List Table

| Node Name                          | Type      | Main Function                                                           | Complexity | Common Use Cases                                           |
| ---------------------------------- | --------- | ----------------------------------------------------------------------- | ---------- | ---------------------------------------------------------- |
| **(Down)load CogVideo Model**      | I/O       | Downloads and loads standard CogVideo models from HuggingFace.          | ★☆☆☆☆      | Obtaining the model before first use.                      |
| **(Down)load CogVideo GGUF Model** | I/O       | Downloads and loads GGUF-formatted CogVideo models.                     | ★☆☆☆☆      | Loading models in environments that only support GGUF.     |
| **(Down)load Tora Model**          | I/O       | Downloads and loads CogVideoX Tora format accelerated models from HuggingFace. | ★☆☆☆☆      | Using Alibaba Tora optimized version for accelerated inference. |
| **CogVideoX Model Loader**         | I/O       | Loads CogVideoX models from a specified path or cache.                  | ★☆☆☆☆      | Custom model management and reuse.                         |
| **CogVideoX VAE Loader**           | I/O       | Loads or switches VAE decoders.                                         | ★☆☆☆☆      | Switching VAEs between different precisions/formats.       |
| **(Down)load CogVideo ControlNet** | I/O       | Downloads and loads CogVideo ControlNet conditional guidance modules.   | ★☆☆☆☆      | Adding ControlNet guidance to video generation; pose/structure-driven generation. |
| **CogVideo ControlNet**            | I/O       | Applies ControlNet conditional network, injecting specific guidance (e.g., pose, depth, bounding box) into the generation process. | ★★☆☆☆      | Pose-driven, structure-driven, style-matching, and other external conditional guidance. |
| **CogVideo TextEncode**            | Process   | Encodes a single text prompt, outputting text latent vectors.           | ★★☆☆☆      | Pre-encoding for text-to-video pipelines.                  |
| **CogVideo DualTextEncode**        | Process   | Encodes positive/negative prompts simultaneously, outputting two sets of text latent vectors. | ★★☆☆☆      | When needing to pass positive and negative prompts for parallel control. |
| **CogVideo TextEncode Combine**    | Process   | Combines multiple text latent vectors, providing a unified input for subsequent sampling. | ★★☆☆☆      | Mixing multiple prompts to generate complex scenes.        |
| **CogVideo ImageEncode**           | I/O       | Encodes static images into spatio-temporal latent vectors usable for video. | ★★★☆☆      | Image encoding in I2V (Image-to-Video) workflows.          |
| **CogVideo ImageEncode FunInP**    | I/O       | Encodes images for Fun-InP (unofficial I2V) models.                   | ★★★☆☆      | Using CogVideoX-Fun for image-to-video.                    |
| **CogVideo Sampler**               | Process   | Performs diffusion sampling of video frames based on text/image latent vectors. | ★★★★☆      | Core sampling for T2V (Text-to-Video) or I2V.              |
| **CogVideo Decode**                | I/O       | Decodes latent vectors back into an image sequence and outputs as video.  | ★★☆☆☆      | Generating the final video output after sampling.          |
| **CogVideoXFun ResizeToClosestBucket** | Process | Automatically adjusts latent vectors to the nearest sampling "bucket" size. | ★☆☆☆☆      | Ensures resolution adjustment compatible with Fun models.    |
| **CogVideo FasterCache**           | Utility   | Enables FasterCache optimization, sacrificing a small amount of VRAM for higher inference speed. | ★★☆☆☆      | Speed improvement for long videos or high-resolution scenes. |
| **CogVideo TorchCompileSettings**  | Utility   | Configures `torch.compile` optimization options.                        | ★★☆☆☆      | Using Triton/SageAttention combinations for compilation acceleration. |
| **CogVideo Context Options**       | Utility   | Sets context window size and FreeNoise noise shuffling strategy.        | ★★☆☆☆      | Vid2Vid or Pose2Vid requiring long sequence context management. |
| **CogVideo LatentPreview**         | Utility   | Previews the effect of intermediate latent vectors in the node panel.     | ★☆☆☆☆      | Debugging or visualizing latent space generation effects.    |
| **CogVideo Enhance-A-Video**       | Process   | Performs post-processing on output video like brightening, denoising, or stylization. | ★★★☆☆      | Post-generation enhancement, e.g., color correction, de-flickering. |
| **CogVideo LoraSelect**            | Advanced  | Dynamically inserts/switches LoRA weights by name or tag in the pipeline. | ★★☆☆☆      | Quickly experimenting with different LoRA effects.         |
| **CogVideo LoraSelect Comfy**      | Advanced  | Seamless integration based on ComfyUI's native LoRA management system.    | ★★★☆☆      | Coordinated use with other ComfyUI LoRA nodes.             |
| **CogVideo TransformerEdit**       | Advanced  | Trims specified Transformer Blocks, removing unnecessary layers to reduce VRAM usage and improve inference efficiency. | ★★★★☆      | Model quantization; experimental layer comparison; accelerated generation in resource-constrained environments. |
| **Tora Encode Trajectory**         | Process   | Uses Tora's trajectory encoder to convert user-drawn motion paths into spatio-temporal motion patch latent vectors. | ★★★★☆      | Precise motion trajectory control in I2V; animation creation. |

## 📑 Core Node Details

### (Down)load CogVideo Model

| Parameter Name | Type    | Default Value  | Property | Description                                     |
| -------------- | ------- | -------------- | -------- | ----------------------------------------------- |
| model_name     | STRING  | "cogvideo-1.1" | Optional | Model name on HuggingFace.                      |
| load_to_vram   | BOOLEAN | True           | Optional | Whether to load directly to VRAM.               |

**Output**
- CogVideo standard format model object.

**Use Cases**
- When running for the first time or if the model is missing, downloads and loads the standard CogVideo model from HuggingFace.

---

### (Down)load CogVideo GGUF Model

| Parameter Name | Type    | Default Value | Property | Description                                     |
| -------------- | ------- | ------------- | -------- | ----------------------------------------------- |
| model_path     | STRING  | —             | Required | Local path or link to the GGUF format model.    |
| quantized      | BOOLEAN | True          | Optional | Whether to load the quantized version.          |

**Output**
- GGUF format compatible model, can be used for inference in low-memory environments.

**Use Cases**
- Loading models in lightweight inference environments that support GGUF, such as webUI plugin deployment.

---

### (Down)load Tora Model

| Parameter Name | Type    | Default Value       | Property | Description                                       |
| -------------- | ------- | ------------------- | -------- | ------------------------------------------------- |
| model_name     | STRING  | "cogvideox-tora"    | Optional | Name of the Tora format optimized model on HuggingFace. |
| load_to_vram   | BOOLEAN | True                | Optional | Whether to load directly to VRAM.                 |

**Output**
- CogVideoX-Tora accelerated model object.

**Use Cases**
- Using the Tora optimized model released by Alibaba for inference performance improvement.

---

### CogVideoX Model Loader

| Parameter Name | Type    | Default Value | Property | Description                                   |
| -------------- | ------- | ------ | -------- | ------------------------------------------- |
| path_or_name   | STRING  | —      | Required | Local path or cached name of the model.       |
| load_to_vram   | BOOLEAN | True   | Optional | Whether to load to GPU.                       |

**Output**
- CogVideoX model object, usable in subsequent ComfyUI workflows.

**Use Cases**
- Managing and loading multiple local models or experimental versions.

---

### CogVideoX VAE Loader

| Parameter Name | Type   | Default Value | Property | Description                                   |
| -------------- | ------ | ---------- | -------- | ------------------------------------------- |
| vae_path       | STRING | —          | Required | Specified VAE weight path.                    |
| precision      | STRING | "fp16"     | Optional | Precision setting, supports "fp16", "bf16", "fp32", etc. |

**Output**
- VAE decoder module, for use with CogVideoX models.

**Use Cases**
- Switching precision or experimenting with different image decoder effects.

---

### (Down)load CogVideo ControlNet

| Parameter Name | Type   | Default Value | Property | Description                                          |
| -------------- | ------ | ---------------- | -------- | -------------------------------------------------- |
| control_type   | STRING | "pose"           | Optional | Control type, e.g., "pose", "depth", "canny", etc. |
| download_url   | STRING | —                | Optional | Custom download link (if not from default model source). |

**Output**
- ControlNet module, can be connected to sampler nodes as a conditional guider.

**Use Cases**
- Introducing structural constraints like pose, outline, depth maps in video generation for fine-grained control.

### CogVideo TextEncode

| Parameter Name | Type    | Default Value | Property | Description                                                         |
| -------------- | ------- | ------ | -------- | ------------------------------------------------------------------- |
| clip           | CLIP    | —      | Required | Provides a CLIP model instance for text encoding, converting prompt text to embedding vectors. |
| prompt         | STRING  | ""     | Required | Input text prompt, used to guide the semantic content of video generation. |
| strength       | FLOAT   | 1.0    | Optional | Controls the strength of text embedding; higher values mean more prominent prompt influence. |
| force_offload  | BOOLEAN | False  | Optional | Whether to load the model on CPU to save GPU VRAM.                  |

**Output**
- `CONDITIONING`: Text conditional embeddings for subsequent sampling.
- `CLIP`: Returns the CLIP model instance for reuse.

**Use Cases**
- Converting natural language prompts into model-recognizable conditional vectors for text-to-video or text-to-image workflows.

---

### CogVideo TextEncode Combine

| Parameter Name | Type | Default Value | Property | Description                                                                 |
| -------------- | ---- | ------ | -------- | --------------------------------------------------------------------------- |
| inputs         | LIST | —      | Required | Multiple text latent vectors from `CogVideo TextEncode` or `DualTextEncode`. |

**Output**
- Combined text latent representation, can be uniformly passed into a Sampler node for video generation.

**Use Cases**
- Fusing multiple prompts to generate complex scene content, controlling the influence of different texts.

---

### CogVideo DualTextEncode

| Parameter Name | Type    | Default Value | Property | Description                                                              |
| -------------- | ------- | --------- | -------- | ------------------------------------------------------------------------ |
| text           | STRING  | —         | Required | Main prompt, controls the primary generated content of the video.        |
| negative_text  | STRING  | ""        | Optional | Negative prompt, used to reduce or exclude undesirable content from the generated video. |
| model          | MODEL   | —         | Required | Loaded `CogVideo` or compatible text encoding model.                     |
| return_dict    | BOOLEAN | True      | Optional | Whether to return encoding results as a dictionary; False returns a single tensor. |

**Output**
- TextEncoderOutput (when return_dict=True): Contains encoded representations of positive and negative text.
- torch.Tensor (when return_dict=False): Encoded text tensor, defaults to positive text latent.

**Use Cases**
- Simultaneously inputting positive and negative prompts for finer generation control, e.g., "an animated scene without watermarks."
- Text guidance for video generation when paired with a Sampler node.

---

### CogVideo Decode

| Parameter Name | Type    | Default Value | Property | Description                                                                 |
| -------------- | ------- | ------ | -------- | --------------------------------------------------------------------------- |
| z              | LATENT  | —      | Required | Input latent video tensor or embedding, output from the sampler node.       |
| return_dict    | BOOLEAN | True   | Optional | Whether to return `DecoderOutput` as a dictionary structure; otherwise, only returns `torch.Tensor`. |

**Output**
- `DecoderOutput` (when `return_dict=True`): Contains decoded video tensor and metadata.
- `torch.Tensor` (when `return_dict=False`): Tensor representation of video frames only.

**Use Cases**
- Decoding latent representations obtained from model inference into visual video frames for saving or re-encoding.

---

### CogVideo Sampler

| Parameter Name      | Type                | Default Value | Property | Description                                                              |
| ------------------- | ------------------- | ------ | -------- | ------------------------------------------------------------------------ |
| model               | COGVIDEOMODEL       | —      | Required | CogVideoX model instance to use for sampling inference.                  |
| positive            | CONDITIONING        | —      | Required | Positive conditional embedding, usually from `CogVideoTextEncode`.       |
| negative            | CONDITIONING        | —      | Optional | Negative conditional embedding, for denoising and contrastive control.   |
| num_frames          | INT                 | —      | Required | Number of video frames to generate.                                      |
| steps               | INT                 | —      | Optional | Sampling steps, affects quality and speed.                               |
| cfg                 | FLOAT               | —      | Optional | Guidance scale; higher values mean stronger reliance on prompts.         |
| seed                | INT                 | —      | Optional | Random seed for result reproducibility.                                  |
| scheduler           | ENUM                | —      | Optional | Scheduler algorithm choice, e.g., DDIM, DPM++, UniPC.                    |
| samples             | LATENT              | —      | Optional | Initial latent tensor for image-to-video or video-to-video workflows.  |
| image_cond_latents  | LATENT              | —      | Optional | Image-encoded latent representation for image-to-video conditioning.   |
| denoise_strength    | FLOAT               | —      | Optional | Denoising strength for video-to-video tasks.                             |
| controlnet          | COGVIDECONTROLNET   | —      | Optional | ControlNet branch input for specific control.                            |
| tora_trajectory     | TORAFEATURES        | —      | Optional | Tora model trajectory features for high-quality motion.                  |
| fastercache         | FASTERCACHEARGS     | —      | Optional | FasterCache configuration for memory/speed balance.                      |
| feta_args           | FETAARGS            | —      | Optional | FreeNoise noise shuffling parameters.                                    |
| teacache_args       | TEACACHEARGS        | —      | Optional | TeaCache parameters for repetitive inference optimization.               |

**Output**
- `LATENT`: Contains the generated video latent representation, needs decoding via `CogVideo Decode`.

**Use Cases**
- Performing text-based, image-based, or existing video-based sampling inference in the graph to generate video latent tensors for text-to-video, image-to-video, or video style transfer, etc.

---

### CogVideo ImageEncode

| Parameter Name | Type    | Default Value | Property | Description                                                                  |
| -------------- | ------- | -------- | -------- | ---------------------------------------------------------------------------- |
| image          | IMAGE   | —        | Required | Static image input to encode, can be single or multiple images.            |
| processor      | VAE     | "auto"   | Optional | VAE encoder type to use; "auto" selects the best VAE based on the model. |
| interpolation  | INT     | 1        | Optional | Image frame interpolation factor for generating smooth transitions.          |
| batch_size     | INT     | 1        | Optional | Number of images to encode simultaneously for batch processing efficiency. |

**Output**
- `LATENT`: Latent tensor sequence corresponding to the image(s), directly usable for video sampling.

**Use Cases**
- Converting static images into latent representations in a video pipeline, supporting image-to-video or image style transfer workflows.

---

### CogVideo ImageEncode FunInP

| Parameter Name | Type    | Default Value | Property | Description                                                                  |
| -------------- | ------- | -------- | -------- | ---------------------------------------------------------------------------- |
| image          | IMAGE   | —        | Required | Static image input to encode for Fun-In-P model workflows.                 |
| fun_model      | STRING  | "Fun-InP"| Required | Specifies the Fun-In-P dedicated model name to use.                        |
| interp_steps   | INT     | 1        | Optional | Number of interpolation frames to generate more intermediate frames during Fun-InP encoding. |
| normalize      | BOOLEAN | True     | Optional | Whether to normalize input images to improve encoding stability.           |

**Output**
- `LATENT_FUN`: Fun-In-P model-specific latent representation, optimized for pose-based or animation-based encoding quality.

**Use Cases**
- In Fun-In-P (pose-driven) video generation or editing workflows, converting static images into latent features with animation potential.

*(The original document seems to have a parameter table for a video loading node here, but it's not explicitly named as a node. I will assume it's a generic video input description or part of another node. If it's a specific node, please clarify its name. For now, I'll translate the parameters as presented.)*

| Parameter Name    | Type   | Default Value | Property | Description                             |
| ----------------- | ------ | ---------- | -------- | --------------------------------------- |
| video_path        | STRING | "input/"   | Required | Video file path.                        |
| force_rate        | FLOAT  | 0.0        | Optional | Force adjust frame rate, 0 to disable.  |
| force_size        | COMBO  | "Disabled" | Optional | Quick resize options.                   |
| frame_load_cap    | INT    | 0          | Optional | Max frames to return (batch size).      |
| skip_first_frames | INT    | 0          | Optional | Skip N frames from the beginning.       |
| select_every_nth  | INT    | 1          | Optional | Sample one frame every N frames.        |

**Output**: IMAGE[] (Image sequence), VHS_VIDEOINFO (Video information), AUDIO (Optional audio)

**Use Cases**:
- Batch processing videos from a server path.
- Processing videos from network URLs.
- Use in automated workflows.

---

### Tora Encode Trajectory

| Parameter Name | Type      | Default Value | Property | Description                                                                 |
| -------------- | --------- | ------ | -------- | --------------------------------------------------------------------------- |
| model          | TORA_MODEL| —      | Required | Input loaded Tora model instance.                                           |
| frames         | IMAGE[]   | —      | Required | Video frame sequence to process for extracting motion trajectory features.  |
| downsample     | INT       | 1      | Optional | Downsampling factor for frame rate or resolution to speed up feature extraction. |
| normalize      | BOOLEAN   | True   | Optional | Whether to normalize input frames to improve trajectory feature stability.    |

**Output**
- `TORA_TRAJECTORY`: Extracted trajectory features for subsequent sampling or control modules.

**Use Cases**
- Providing motion trajectory conditions to CogVideoSampler in video-to-video or text-to-video generation pipelines for more coherent animation.

---

### CogVideo LoraSelect

| Parameter Name | Type          | Default Value | Property | Description                                                                  |
| -------------- | ------------- | ------ | -------- | ---------------------------------------------------------------------------- |
| model          | COGVIDEOMODEL | —      | Required | Current base CogVideoX model instance being used.                            |
| lora_name      | STRING        | —      | Required | Name of the LoRA weights to load (e.g., "lora-style").                     |
| lora_scale     | FLOAT         | 1.0    | Optional | Strength ratio of LoRA weights applied to the base model.                  |
| merge          | BOOLEAN       | False  | Optional | Whether to permanently merge LoRA weights into the base model.             |

**Output**
- `MODIFIED_MODEL`: Model instance with LoRA weights applied, ready for sampling.

**Use Cases**
- Dynamically selecting and applying LoRA weights for generating videos with specific styles or enhanced details without restarting or reloading the main model.

---

### CogVideo ControlNet

| Parameter Name | Type                | Default Value | Property | Description                                                                 |
| -------------- | ------------------- | ------ | -------- | --------------------------------------------------------------------------- |
| model          | COGVIDEOCONTROLNET  | —      | Required | Input ControlNet model instance.                                            |
| conditioning   | CONDITIONING        | —      | Required | Conditional embedding to apply (e.g., from `CogVideoTextEncode` or `CogVideoImageEncode`). |
| weight         | FLOAT               | 1.0    | Optional | Weight ratio controlling the influence of ControlNet.                       |
| start_step     | INT                 | 0      | Optional | Starting step in the sampling process for application.                      |
| end_step       | INT                 | —      | Optional | Ending application step in sampling (defaults to the last step).            |

**Output**
- `MODIFIED_CONDITIONING`: Conditional embedding after applying ControlNet, can be passed to a sampler node.

**Use Cases**
- Using ControlNet in conjunction with the main model when structured or temporal constraints need to be applied to the generation process.

---

### CogVideoXFun ResizeToClosestBucket

| Parameter Name      | Type        | Default Value | Property | Description                                                                                      |
| ------------------- | ----------- | ------ | -------- | ------------------------------------------------------------------------------------------------ |
| `latents`           | LATENT      | —      | Required | Input video latent vectors, possibly not matching model's required resolution or frame count.    |
| `bucket_heights`    | LIST[int]   | —      | Required | List of supported height "buckets", e.g., `[256, 384, 512, 640]`.                                |
| `bucket_widths`     | LIST[int]   | —      | Required | List of supported width "buckets", must correspond one-to-one with `bucket_heights` indices.     |
| `num_frames`        | INT         | —      | Optional | Optionally provide a frame count bucket list (frame counts compatible with the model), otherwise only spatial dimensions are adjusted. |
| `mode`              | STRING      | `pad`  | Optional | Spatial adjustment mode: `pad` (zero-padding to bucket size), or `crop` (center cropping to bucket size). |
| `align_to_bucket`   | BOOLEAN     | True   | Optional | Whether to strictly align to the nearest bucket size; if `False`, takes the closest bucket not exceeding the current size. |

**Output**
- Returns `ResizedLatents`: Video latent tensor adjusted to align with specified "bucket" dimensions, shaped `[batch, num_frames, channels, target_height, target_width]`.

**Use Cases**
- Preprocessing latent vectors before using Fun-InP or other models with strict "bucket" requirements for resolution/frame count.
- Used with `CogVideoXFun Sampler` to ensure input dimensions are valid and correctly handled by the scheduler, improving compatibility and avoiding errors.

---

### CogVideo FasterCache

| Parameter Name    | Type    | Default Value | Property | Description                                                                  |
| ----------------- | ------- | ------ | -------- | ---------------------------------------------------------------------------- |
| `enable_cache`    | BOOLEAN | True   | Optional | Whether to enable caching mechanism to reuse intermediate computation results and reduce redundant calculations. |
| `cache_size_mb`   | INT     | 512    | Optional | Maximum cache size (MB); older cache is cleared using LRU strategy when exceeded. |
| `cache_dtype`     | STRING  | `"fp16"`| Optional | Data type for cached tensors, options include `"fp16"`, `"fp32"`, balancing speed and precision. |

**Output**
- `cached`: Boolean, indicates if the current call hit the cache.
- `cache_info`: Dictionary, contains statistics like current cache usage and hit rate.

**Use Cases**
- For repeatedly calling large model inference under the same conditions, caching intermediate activations or weight compilation results can significantly improve overall pipeline speed.

---

### CogVideo TorchCompileSettings

| Parameter Name       | Type    | Default Value | Property | Description                                                                  |
| -------------------- | ------- | -------- | -------- | ---------------------------------------------------------------------------- |
| `compile_mode`       | STRING  | `"default"`| Optional | `torch.compile` mode, options: `"default"`, `"reduce-overhead"`, `"max-autotune"`, affecting compilation optimization strength. |
| `backend`            | STRING  | `"inductor"`| Optional | Compilation backend, options: `"inductor"`, `"nvfuser"`, determining underlying code generation. |
| `autotune`           | BOOLEAN | False    | Optional | Whether to enable auto-tuning, trying multiple compilation strategies and choosing the best; adds extra time at startup. |

**Output**
- `compile_settings`: Dictionary, returns the actual effective compile mode, backend, and auto-tuning status.

**Use Cases**
- Accelerating model execution via PyTorch 2.0 compilation in scenarios with ample GPU resources and extreme demand for inference speed; can also switch backends for different hardware.

---

### CogVideo Context Options

| Parameter Name       | Type    | Default Value | Property | Description                                                                  |
| -------------------- | ------- | ------------ | -------- | ---------------------------------------------------------------------------- |
| `context_window`     | INT     | 32           | Optional | Maximum temporal context length (frames); earliest frames are discarded if exceeded. |
| `free_noise_stride`  | INT     | 4            | Optional | When executing free_noise mechanism, how many frames to skip as perturbation, controlling noise shuffling granularity. |
| `enable_pos_embed`   | BOOLEAN | True         | Optional | Whether to add positional embeddings to temporal frames, improving sequential information representation. |

**Output**
- `context_config`: Dictionary, contains the final context window, stride, and positional embedding status.

**Use Cases**
- In long video (Vid2Vid, Pose2Vid) generation, limiting context length to save VRAM, while adding perturbations between keyframes to improve coherence.

---

### CogVideo LatentPreview

| Parameter Name     | Type  | Default Value | Property | Description                                                                  |
| ------------------ | ----- | ------ | -------- | ---------------------------------------------------------------------------- |
| `preview_frame`    | INT   | 0      | Optional | Frame index to visualize (0 to num_frames-1); displays its latent map in the node panel. |
| `scale_factor`     | FLOAT | 2.0    | Optional | Upsamples the latent map for clearer preview in the panel.                   |

**Output**
- `preview_image`: PIL Image object, displays the visual latent map of the corresponding frame in the UI panel.

**Use Cases**
- During debugging of intermediate generation stages, quickly checking the effect of a single frame's latent vector to help adjust sampling or model parameters.

---

### CogVideo Enhance-A-Video

| Parameter Name     | Type   | Default Value | Property | Description                                                                  |
| ------------------ | ------ | ------ | -------- | ---------------------------------------------------------------------------- |
| `denoise_strength` | FLOAT  | 0.5    | Optional | Post-processing denoise strength (0.0–1.0); higher values denoise more but may lose detail. |
| `color_boost`      | FLOAT  | 1.2    | Optional | Color enhancement factor for fine-tuning output video colors, increasing saturation and contrast. |
| `apply_style`      | STRING | `None` | Optional | Optional style preset name (e.g., `"cinematic"`, `"soft"`) for stylizing the video. |

**Output**
- `enhanced_video`: Post-processed video tensor or file path, ready for saving or further encoding.

**Use Cases**
- After the generation process, denoising, color grading, or applying a unified style to the video for a more visually appealing final result.

---

### CogVideo LoraSelect (Duplicate, likely refers to the general LoRA selection)

| Parameter Name | Type   | Default Value | Property | Description                                     |
| -------------- | ------ | ------ | -------- | ----------------------------------------------- |
| `model`        | MODEL  | —      | Required | Input CogVideo model object.                    |
| `lora_path`    | STRING | —      | Required | Path to the LoRA weight file to load.           |
| `strength`     | FLOAT  | 1.0    | Optional | LoRA weight application strength (0.0–1.0).   |

**Output**
- CogVideo model object with the specified LoRA weights applied.

**Use Cases**
- Dynamically replacing or overlaying LoRA weights in video generation for style fine-tuning or effect enhancement.

---

### CogVideo LoraSelect Comfy

| Parameter Name | Type   | Default Value | Property | Description                                                                  |
| -------------- | ------ | ------ | -------- | ---------------------------------------------------------------------------- |
| `model`        | MODEL  | —      | Required | Input CogVideo model object.                                                 |
| `lora_name`    | STRING | —      | Required | Pre-set LoRA name (weights already in repository or ComfyUI directory).    |
| `strength`     | FLOAT  | 1.0    | Optional | LoRA weight application strength (0.0–1.0), controlling rendering effect proportion. |

**Output**
- Model object with LoRA weights loaded, compatible with ComfyUI's native system.

**Use Cases**
- Seamlessly calling ComfyUI's LoRA management resources to quickly switch between different preset weights.

---

### CogVideo TransformerEdit

| Parameter Name    | Type   | Default Value | Property | Description                                                                  |
| ----------------- | ------ | ------ | -------- | ---------------------------------------------------------------------------- |
| `remove_blocks`   | STRING | ""     | Required | Comma-separated list of Transformer Block indices, e.g., `"15,25,37"`.      |

**Output**
- `block_list`: Integer list, actual removed Block indices.

**Use Cases**
- Precisely trimming model layers to reduce VRAM footprint or accelerate inference; for experimental layer comparison.

---

### Tora Encode Trajectory (Duplicate, likely refers to the trajectory encoding part)

| Parameter Name      | Type    | Default Value | Property | Description                                                                  |
| ------------------- | ------- | ---------- | -------- | ---------------------------------------------------------------------------- |
| `trajectory`        | PATH    | —          | Required | User-drawn or imported motion trajectory file (SVG, JSON, etc.).           |
| `resolution`        | STRING  | `"512x512"`| Optional | Spatial resolution (width × height) for generated latent patches.          |
| `frame_count`       | INTEGER | 16         | Optional | Number of motion patch frames to generate.                                   |

**Output**
- `trajectory_embeds`: Spatio-temporal motion patch latent vectors, directly usable in sampler nodes.

**Use Cases**
- Controlling motion according to custom trajectories in I2V generation flows; suitable for animation and path-driven effects.

## 🔧 Common Workflow Combinations

| Workflow Name                  | Node Combination                                        | Purpose                                                                      |
|--------------------------------|---------------------------------------------------------|------------------------------------------------------------------------------|
| Text-to-Video Generation       | CogVideo TextEncode → CogVideo Sampler → CogVideo Decode | Generates video content based on text prompts (core function, highest stability). |
| Image-to-Video Conversion      | CogVideo ImageEncode → CogVideo Sampler → CogVideo Decode | Extends static images into dynamic videos (requires verifying image encoder compatibility). |
| Video Generation with Motion Trajectory Control | Tora Encode Trajectory → CogVideo Sampler → CogVideo Decode | Controls object motion paths via trajectory coordinates (depends on trajectory encoder precision). |
| Accelerated Video Generation   | CogVideo FasterCache → CogVideo Sampler → CogVideo Decode | Improves generation speed through VRAM optimization (tested speedup of 20%-30%). |
| Context Window Adjustment      | CogVideo Context Options → CogVideo Sampler → CogVideo Decode | Adjusts temporal context length (effective in the 16-64 frame range).          |
| Video Enhancement Processing   | CogVideo Decode → CogVideo Enhance-A-Video              | Resolution upscaling/frame interpolation enhancement (requires separate deployment of enhancement module). |

---