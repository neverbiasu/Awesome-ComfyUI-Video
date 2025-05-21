# 🎬 HunyuanVideoWrapper Nodes

## 🔍 Overview

| Item         | Details                                                            |
| ------------ | ------------------------------------------------------------------ |
| 📌 **Author** | [@Kijai](https://github.com/Kijai)                                 |
| 📅 **Version** | 1.0.0+                                                           |
| 🏷️ **Category** | Video Loading & Processing, Image Effects, Model Utilities       |
| 🔗 **Repository** | [GitHub Repository Link](https://github.com/Kijai/HunyuanVideoWrapper) |

## 📝 Introduction
The `HunyuanVideoWrapper` provides a comprehensive set of nodes for video generation, processing, and enhancement. It includes tools for sampling, encoding, decoding, and advanced features like context management, CFG, LoRA integration, and inverse sampling.

## 📊 Node List

| Node Name                             | Main Function                                                          | Category                 | Frequency | Common Use Case                            | Recommendation | Role in Workflow                                        |
| ------------------------------------- | ---------------------------------------------------------------------- | ------------------------ | --------- | ------------------------------------------ | -------------- | ------------------------------------------------------- |
| **DownloadAndLoadHyVideoTextEncoder** | Download & load LLM or CLIP text encoders for conditioning             | I/O Loader               | 7         | Text embedding input for prompt pipelines  | ★★★★☆          | Supplies `HYVIDTEXTENCODER` to `TextEncode` nodes       |
| **HyVideoTextEncode**                 | Encode text prompts into conditioning embeddings                       | Conditioning Encoder     | 7         | Text-to-video prompt embedding             | ★★★★☆          | Converts language prompts into model-consumable vectors |
| **HyVideoModelLoader**                | Load core model: UNet, scheduler, LoRA, attention, precision           | I/O Loader               | 7         | Initialize generation pipeline             | ★★★★★          | Supplies `HYVIDEOMODEL` to sampler nodes                |
| **HyVideoVAELoader**                  | Load 3D VAE for latent-image-frame conversions                         | I/O Loader               | 7         | Encode/decode latent frames                | ★★★★★          | Supplies `VAE` to encoder/decoder nodes                 |
| **HyVideoSampler**                    | Core denoising sampler over space-time latent video                    | Generative Sampler       | 7         | Text-to-video / Image-to-video             | ★★★★★          | Produces final `LATENT` video tensor                    |
| **HyVideoDecode**                     | Decode latent video into image frames                                  | Decoder                  | 7         | Render output frames                       | ★★★★★          | Visual output for generated video                       |
| **HyVideoBlockSwap**                  | Swap UNet blocks or offload to CPU for optimization                    | Architectural Control    | 5         | Hybrid CPU/GPU or structure experiments    | ★★★★☆          | Supplies `BLOCKSWAPARGS` to model loader                |
| **HyVideoTorchCompileSettings**       | Generate `torch.compile` settings to boost speed and reduce VRAM       | Performance Optimizer    | 4         | CUDA graph compilation                     | ★★★★☆          | Links with `ModelLoader` to enable compilation          |
| **HyVideoTeaCache**                   | Cache transformer activations to save memory during sampling           | Caching Optimizer        | 4         | Long video inference                       | ★★★★☆          | Provides `TEACACHEARGS` to sampler                      |
| **HyVideoEncode**                     | Encode single frame into latent space                                  | Encoder                  | 3         | I2V workflows                              | ★★★☆☆          | Converts input image to latent format                   |
| **HyVideoEnhanceAVideo**             | Generate denoise/upscale parameters to improve output                  | Post-processing          | 3         | Long sequence enhancement                  | ★★★☆☆          | Applies detail enhancement to final video               |
| **HyVideoLoraSelect**                 | Load and blend LoRA weights with strength/masks                        | LoRA Manager             | 3         | Style or character control                 | ★★★☆☆          | Supplies `HYVIDLORA` to model loader                    |
| **HyVideoI2VEncode**                  | Encode text + optional image into video embedding                      | Conditioning Encoder     | 2         | Image-to-video workflows                   | ★★★☆☆          | Produces multimodal video embeddings                    |
| **HyVideoEncodeKeyframes**           | Encode start/end keyframes into latent space                           | Encoder                  | 2         | Keyframe-guided interpolation              | ★★★☆☆          | Enables pose or storyboard interpolation                |
| **HyVideoGetClosestBucketSize**       | Calculate closest model-supported resolution bucket                    | Resolution Tool          | 2         | Resize input to valid dimensions           | ★★★☆☆          | Outputs valid width/height                              |
| **HyVideoCFG**                        | Build classifier-free guidance (CFG) schedule and weights              | CFG Manager              | 1         | Dynamically adjust prompt influence        | ★★★☆☆          | Controls unconditional branch weight over time          |
| **HyVideoTextImageEncode**           | Encode two images and text into a fused conditioning embedding         | Conditioning Encoder     | 1         | IP2V (image + prompt to video)             | ★★★★☆          | Builds multimodal prompt embeddings                     |
| **HyVideoInverseSampler**            | Reverse-sample latent back to noise                                    | Reverse Sampler          | 1         | Latent editing or inversion                | ★★★★☆          | Provides noise for editable re-generation               |
| **HyVideoReSampler**                 | Resample latent with updated seed, CFG, or motion                      | Resampler                | 1         | Post-refinement of motion                  | ★★★☆☆          | Applies new settings without full reinitialization      |
| **HyVideoPromptMixSampler**         | Mix multiple embeddings over time using interpolation curves           | Mixed Sampler            | 1         | Prompt/style blending and transition       | ★★★★☆          | Smooth transitions between prompts                      |
| **HyVideoLoraBlockEdit**             | Enable/disable specific LoRA blocks                                    | LoRA Tweaker             | 1         | Fine-grained LoRA blending                 | ★★★☆☆          | Refines LoRA injection per-block                        |
| **SplineEditor**                      | Visual editor for curve interpolation of weights                       | Interpolation Curve Tool | 1         | Prompt transitions                         | ★★★☆☆          | Used with `PromptMixSampler` to control transitions     |
| **HyVideoEnableAVideo**             | Configure weighted Enhance-A-Video parameters                          | Post-processing          | 1         | Used with `EnhanceAVideo`                  | ★★★☆☆          | —                                                       |
| **HyVideoLatentPreview**             | Preview intermediate latent video for debugging                        | Visualization            | 1         | Debugging latent stages                    | ★☆☆☆☆          | Requires VHS latent preview enabled                     |
| **SetNode / GetNode / INTConstant**  | Set/get global variables (e.g., frame count, index range)              | Utility                  | Many      | Cross-node parameter sharing               | ★☆☆☆☆          | Manage parameters across disconnected nodes             |
| **HyVideoCustomPromptTemplate**      | Custom template for prompt formatting                                  | Utility                  | 1         | Structured prompt control                  | ★☆☆☆☆          | Used as input to `TextEncode`                           |
| **HyVideoSTG**                        | Generate spatio-temporal guidance arguments                            | Motion Control           | 1         | Motion consistency, masked flows           | ★★★★☆          | Supplies `STGARGS` to sampler                           |
| **HunyuanVideoFresca**              | Experimental frequency-based motion modulation (FreSca)                | Experimental             | 1         | Spectral style blending                    | ★★★☆☆          | Supplies `FRESCAARGS` for denoising or output stages    |
| **HunyuanVideoSLG**                  | Selectively skip unconditional guidance on UNet blocks                 | Experimental             | 1         | UNet ablation and testing                  | ★★☆☆☆          | Supplies `SLGARGS` to sampler or model loader           |
| **HyVideoLoopArgs**                  | Create loop arguments for cyclic latent transformations                | Loop Utility             | 1         | Seamless looping or morphing effects       | ★★☆☆☆          | Provides `LOOPARGS` to sampler                          |
| **HyVideoContextOptions**            | Manage context window size/stride for long sequence generation         | Context Management       | 1         | Long video with frame overlap              | ★★★☆☆          | Provides `CTXARGS` to sampler                           |

## 📑 Core Node Details

### DownloadAndLoadHyVideoTextEncoder

| **Parameter Name**          | **Type**                        | **Default Value**                              | **Property** | **Description**                                                                 |
|-----------------------------|---------------------------------|------------------------------------------|-------------|---------------------------------------------------------------------------------|
| `llm_model`               | COMBO<sup>1</sup>               | "Kijai/llava-llama-3-8b-text-encoder-tokenizer" | Required    | Large Language Model (LLM) for text encoding                                     |
| `clip_model`              | COMBO<sup>2</sup>               | "disabled"                              | Optional    | CLIP vision model (optional for multimodal encoding)                            |
| `precision`               | COMBO<sup>3</sup>               | "bf16"                                 | Optional    | Floating-point precision mode                                                   |
| `apply_final_norm`        | BOOLEAN                         | False                                  | Optional    | Apply final normalization layer to text embeddings                              |
| `hidden_state_skip_layer` | INT                             | 2                                      | Optional    | Skip hidden states from the specified layer (reduces computation)               |
| `quantization`            | COMBO<sup>4</sup>               | "disabled"                             | Optional    | Quantization method for memory optimization                                     |
| `load_device`             | COMBO                           | "offload_device"                       | Optional    | Device to load the text encoder (GPU or CPU)                                    |

**Output:** `HYVIDTEXTENCODER`  

**Use Cases:**  
- **Text-to-Video** workflows requiring advanced language understanding  
- Multimodal encoding (text + image) when `clip_model` is enabled  
- Low-resource environments via `quantization` and `offload_device`  

---

### HyVideoTextEncode

| **Parameter Name**          | **Type**                        | **Default Value**                              | **Property** | **Description**                                                                 |
| ----------------------- | --------------------- | ----------- | ------------- | ----------------------------------------------------------------------- |
| `text`                  | `STRING`              | —           | Required      | Text prompt to be embedded                                              |
| `text_encoder`          | `HYVIDTEXTENCODER`    | —           | Required      | Text encoder object loaded from `DownloadAndLoadHyVideoTextEncoder`     |
| `is_negative_prompt`    | `BOOLEAN`             | `False`     | Optional      | Marks the prompt as a negative condition (for classifier-free guidance) |
| `custom_prompt_context` | `CustomPromptContext` | —           | Optional      | Structured prompt template support for LLM use                          |

**Output**:`HYVIDEMBEDS` – Encoded embeddings used as conditioning input

**Use Cases**:
- Standard prompt encoding for **text-to-video** workflows
- Used in both positive and negative conditioning branches
- Supports **custom prompt templates** via prompt formatting

---

### HyVideoModelLoader

| **Parameter Name**          | **Type**                        | **Default Value**                              | **Property** | **Description**                                                                 |
|----------------------|-------------------|--------------------|-----------|-----------------------------------------------------------------------------|
| `model`              | `folder_paths list` | —                 | Required  | Path(s) under `ComfyUI/models/diffusion_models`                             |
| `base_precision`     | COMBO             | `bf16`             | Required  | Precision type used for base weights                                        |
| `quantization`       | COMBO             | `disabled`         | Required  | Apply 8-bit quantization to weights or keep in full precision              |
| `load_device`        | COMBO             | `main_device`      | Required  | Target device for loading: GPU or CPU                                       |
| `attention_mode`     | COMBO             | `flash_attn_varlen`| Optional  | Type of attention implementation to use                                    |
| `compile_args`       | `COMPILEARGS`     | —                  | Optional  | Optional Torch compile acceleration settings                               |
| `block_swap_args`    | `BLOCKSWAPARGS`   | —                  | Optional  | Used to offload or replace blocks (see `HyVideoBlockSwap`)                |
| `lora`               | `HYVIDLORA`       | `None`             | Optional  | LoRA weight file with strength and block masking                           |
| `auto_cpu_offload`   | BOOLEAN           | `False`            | Optional  | Automatically offload unused modules to CPU                                |
| `upcast_rope`        | BOOLEAN           | `True`             | Optional  | Use float32 for rotary positional embeddings for improved numerical stability |

**Output**:**`HYVIDEOMODEL`**-The fully initialized model object required by `HyVideoSampler` and other generation nodes.

**Use Cases**:
- Always the first node in any HunyuanVideo generation pipeline  
- Swap different UNet checkpoints dynamically  
- Integrate LoRA, quantization, and block offloading  
- Switch attention backend for speed/quality trade-offs  
- Activate `torch.compile` and memory optimization features  

--- 

### HyVideoVAELoader

| **Parameter Name**          | **Type**                        | **Default Value**                              | **Property** | **Description**                                                                 |
| -------------- | ------------------------- | ----------- | ------------- | ----------------------------------------------- |
| `model_name`   | `folder_paths list (vae)` | —           | Required      | Select from models under `ComfyUI/models/vae`   |
| `precision`    | `fp16`, `fp32`, `bf16`    | `bf16`      | Optional      | Floating-point precision for inference          |
| `compile_args` | `COMPILEARGS`             | —           | Optional      | Optional compile settings using `torch.compile` |

**Output:** `VAE`

**Use Cases:**
* Used for **encoding** input images into latent space
* Used for **decoding** latent output back to video frames
* Paired with encoder/sampler/decoder nodes

---

## HyVideoSampler

| **Parameter Name**          | **Type**                        | **Default Value**                              | **Property** | **Description**                                                                 |
| ------------------------- | ---------------------------- | ---------------------------- | ------------- | ----------------------------------------------------------------- |
| `model`                   | `HYVIDEOMODEL`               | —                            | Required      | The core model loaded via `HyVideoModelLoader`.                   |
| `hyvid_embeds`            | `HYVIDEMBEDS`                | —                            | Required      | Conditioning embeddings generated from prompt encoders.           |
| `width`                   | `INT`                        | `512`                        | Required      | Output video frame width (must be divisible by 16).               |
| `height`                  | `INT`                        | `512`                        | Required      | Output video frame height (must be divisible by 16).              |
| `num_frames`              | `INT`                        | `49`                         | Required      | Number of video frames. Must satisfy `(num_frames - 1) % 4 == 0`. |
| `steps`                   | `INT`                        | `30`                         | Required      | Denoising steps used in sampling loop.                            |
| `embedded_guidance_scale` | `FLOAT`                      | `6.0`                        | Required      | CFG strength for the prompt embeddings.                           |
| `flow_shift`              | `FLOAT`                      | `9.0`                        | Required      | Controls intensity of temporal flow (motion guidance).            |
| `seed`                    | `INT`                        | `0`                          | Required      | Random seed for reproducibility.                                  |
| `force_offload`           | `BOOLEAN`                    | `True`                       | Required      | Whether to offload the model after sampling to free GPU memory.   |
| `samples`                 | `LATENT`                     | `None`                       | Optional      | Input latent for video-to-video or re-generation.                 |
| `image_cond_latents`      | `LATENT`                     | `None`                       | Optional      | Initial latent derived from image encoding (for I2V).             |
| `denoise_strength`        | `FLOAT`                      | `1.0`                        | Optional      | Used for controlling interpolation strength during I2V workflows. |
| `stg_args`                | `STGARGS`                    | —                            | Optional      | Spatio-temporal guidance arguments from `HyVideoSTG`.             |
| `context_options`         | `HYVIDCONTEXT`               | —                            | Optional      | Configuration for transformer context window.                     |
| `feta_args`               | `FETAARGS`                   | —                            | Optional      | Feature enhancement settings from `HyVideoEnhanceAVideo`.         |
| `teacache_args`           | `TEACACHEARGS`               | —                            | Optional      | Transformer activation cache configuration.                       |
| `scheduler`               | `Enum`                       | `FlowMatchDiscreteScheduler` | Optional      | Sampling algorithm scheduler used during denoising.               |
| `riflex_freq_index`       | `INT`                        | `0`                          | Optional      | Controls RIFLEX interpolation frequency (0 disables RIFLEX).      |
| `i2v_mode`                | `"stability"` or `"dynamic"` | `dynamic`                    | Optional      | I2V mode: `dynamic` adapts motion, `stability` reduces variance.  |
| `loop_args`               | `LOOPARGS`                   | —                            | Optional      | Loop consistency configuration (for cyclic generation).           |
| `fresca_args`             | `FRESCA_ARGS`                | —                            | Optional      | Frequency-based motion control parameters (FreSca).               |
| `slg_args`                | `SLGARGS`                    | —                            | Optional      | Selective latent guidance arguments (from `HunyuanVideoSLG`).     |
| `mask`                    | `MASK`                       | —                            | Optional      | Pixel/frame mask for partial or inpainting-style generation.      |

**Output**: **`samples`** (`LATENT`) – The final latent video tensor, to be decoded using `HyVideoDecode`.

**Use Cases**:
* Core engine for all **text-to-video**, **image-to-video**, and **video-to-video** workflows
* Accepts external latents for **style transfer**, **remixing**, or **conditional continuation**
* Works with optional modules: `STG`, `TeaCache`, `FETA`, `FreSca`, `LoopArgs`, `SLG`, `RIFLEX`
* Ideal for long, high-resolution sequences with adaptive memory-saving options

---

### HyVideoDecode

| **Parameter Name**          | **Type**                        | **Default Value**                              | **Property** | **Description**                                                                 |
| ------------------------------ | --------- | ----------- | ------------- | ---------------------------------------------------------- |
| `vae`                          | `VAE`     | —           | Required      | The VAE model loaded from `HyVideoVAELoader`               |
| `samples`                      | `LATENT`  | —           | Required      | Latent video tensor to decode                              |
| `enable_vae_tiling`            | `BOOLEAN` | `True`      | Required      | Enable tiling to reduce VRAM usage                         |
| `temporal_tiling_sample_size`  | `INT`     | `64`        | Required      | Latent-frame tiling size (smaller = less VRAM, more seams) |
| `spatial_tile_sample_min_size` | `INT`     | `256`       | Required      | Minimum spatial tile size in pixels                        |
| `auto_tile_size`               | `BOOLEAN` | `True`      | Required      | Ignore above and use default tiling                        |
| `skip_latents`                 | `INT`     | `0`         | Optional      | Skip frames from beginning of latent input                 |
| `balance_brightness`           | `BOOLEAN` | `False`     | Optional      | Attempt to correct brightness between tiles                |

**Output**: `IMAGE`

**Use Cases**:
* Required to convert final latent tensor into **visual frames**
* Used after `HyVideoSampler` in most workflows
* Useful for long videos with low memory via **tiling**

---

### HyVideoBlockSwap

| **Parameter Name**          | **Type**                        | **Default Value**                              | **Property** | **Description**                                                                 |
| ------------------------ | -------- | ----------- | ------------- | ------------------------------------------ |
| double\_blocks\_to\_swap | INT      | 20          | Required      | Number of double blocks to offload to CPU. |
| single\_blocks\_to\_swap | INT      | 0           | Required      | Number of single blocks to offload to CPU. |
| offload\_txt\_in         | BOOLEAN  | False       | Required      | Whether to offload the `txt_in` layer.     |
| offload\_img\_in         | BOOLEAN  | False       | Required      | Whether to offload the `img_in` layer.     |

**Output**: `BLOCKSWAPARGS`

**Use Cases**:
* Use with `HyVideoModelLoader` to reduce VRAM usage by partial CPU offload.
* Common in low-memory environments or with large batch sizes.

---

### HyVideoTorchCompileSettings

| **Parameter Name**          | **Type**                        | **Default Value**                              | **Property** | **Description**                                                                 |
|--------------------------------|---------------------|-------------|-------------|---------------------------------------------------------------------------------|
| `backend`                    | COMBO              | "inductor"  | Required    | Torch compilation backend (`inductor` or `cudagraphs`)                          |
| `fullgraph`                  | BOOLEAN           | False       | Required    | Enforce full-graph compilation (better optimization but lower compatibility)     |
| `mode`                       | COMBO<sup>1</sup> | "default"   | Required    | Compilation optimization mode                                                   |
| `dynamic`                    | BOOLEAN           | False       | Required    | Enable dynamic shape support (optimized for long sequences)                      |
| `dynamo_cache_size_limit`    | INT               | 64          | Required    | Dynamic compilation cache size (in MB)                                          |
| `compile_single_blocks`      | BOOLEAN           | True        | Required    | Compile single-block modules                                                    |
| `compile_double_blocks`      | BOOLEAN           | True        | Required    | Compile double-block modules                                                    |
| `compile_txt_in`             | BOOLEAN           | False       | Required    | Compile text input layers                                                       |
| `compile_vector_in`          | BOOLEAN           | False       | Required    | Compile vector input layers                                                     |
| `compile_final_layer`        | BOOLEAN           | False       | Required    | Compile final output layer                                                      |

**Output**: `COMPILEARGS`  

**Use Cases**:  
- Maximize inference speed (e.g., real-time generation)  
- VRAM optimization for long video generation (>100 frames)  

---

### HyVideoTeaCache

| **Parameter Name**          | **Type**                        | **Default Value**                              | **Property** | **Description**                                                                 |
| --------------- | ----------------------------------- | ----------------- | ------------- | --------------------------------------------------------------------- |
| rel\_l1\_thresh | FLOAT                               | 0.15              | Required      | Aggressiveness of caching; higher = faster but more risk of artifacts |
| cache\_device   | `["main_device", "offload_device"]` | "offload\_device" | Required      | Device to cache transformer activations                               |
| start\_step     | INT                                 | 0                 | Required      | Start sampling step to apply TeaCache                                 |
| end\_step       | INT                                 | -1                | Required      | Final step to apply TeaCache (`-1` = till end)                        |

**Output:** `TEACACHEARGS`

**Use Cases:**
* Speeds up sampling by **caching transformer activations**
* Useful for **long video generations** where recomputation is expensive
* Can **reduce VRAM usage** when paired with offload device

---

### HyVideoEncode

| **Parameter Name**          | **Type**                        | **Default Value**                              | **Property** | **Description**                                                                 |
|--------------------------------|---------------------|-------------|-------------|---------------------------------------------------------------------------------|
| `vae`                        | VAE               | —           | Required    | VAE model loaded via `HyVideoVAELoader`                                        |
| `image`                      | IMAGE             | —           | Required    | Input image tensor (B x H x W x C)                                              |
| `enable_vae_tiling`          | BOOLEAN           | True        | Required    | Enable tiling to reduce VRAM usage (may introduce minor seams)                  |
| `temporal_tiling_sample_size`| INT               | 64          | Required    | Temporal tile size (must be 64 for smooth operation)                            |
| `spatial_tile_sample_min_size`| INT               | 256         | Required    | Minimum spatial tile size in pixels (smaller values save VRAM)                  |
| `auto_tile_size`             | BOOLEAN           | True        | Required    | Auto-configure tile sizes based on defaults                                     |
| `noise_aug_strength`         | FLOAT             | 0.0         | Optional    | Add noise to input image (enhances motion, range: 0.0-10.0)                     |
| `latent_strength`            | FLOAT             | 1.0         | Optional    | Scale factor for latent output (lower values allow more motion)                 |
| `latent_dist`                | COMBO<sup>1</sup> | "sample"    | Optional    | Latent sampling mode (`sample`/`mode`)                                         |

**Output:** `LATENT`  

**Use Cases:**  
- **Image-to-Video** workflows (e2v)  
- Preprocessing reference images for **keyframe-based generation**  
- Adding noise augmentation to induce motion  

---

### HyVideoEnhanceAVideo

| **Parameter Name**          | **Type**                        | **Default Value**                              | **Property** | **Description**                                                                 |
| -------------- | -------- | ----------- | ------------- | ------------------------------------------------------ |
| weight         | FLOAT    | 2.0         | Required      | FETA weight to control enhancement strength.           |
| single\_blocks | BOOLEAN  | True        | Required      | Whether to enable Enhance-A-Video for single blocks.   |
| double\_blocks | BOOLEAN  | True        | Required      | Whether to enable Enhance-A-Video for double blocks.   |
| start\_percent | FLOAT    | 0.0         | Required      | Start percent of denoising steps to apply enhancement. |
| end\_percent   | FLOAT    | 1.0         | Required      | End percent of denoising steps to apply enhancement.   |

**Output:** `FETAARGS`

**Use Cases:**
* Used as input to the `HyVideoSampler` node to boost detail.
* Effective for sharpening and denoising long sequences.

---

### HyVideoLoraSelect

| **Parameter Name**          | **Type**                        | **Default Value**                              | **Property** | **Description**                                                                 |
| ------------- | -------------- | ----------- | ------------- | ------------------------------------------------------------------ |
| lora          | file name list | —           | Required      | Path to LoRA weight file (`.safetensors`) in `models/loras` folder |
| strength      | FLOAT          | `1.0`       | Required      | Strength of LoRA injection (set to 0.0 to unmerge)                 |
| prev\_lora    | HYVIDLORA      | `None`      | Optional      | Chain multiple LoRAs                                               |
| blocks        | SELECTEDBLOCKS | `None`      | Optional      | Selective block injection via `HyVideoLoraBlockEdit`               |

**Output:** `HYVIDLORA`

**Use Cases:**
* Inject style/personality/character behavior into the UNet
* Combine multiple LoRAs with custom strength

---

### HyVideoI2VEncode

| **Parameter Name**          | **Type**                        | **Default Value**                              | **Property** | **Description**                                                                 |
|--------------------------------|---------------------|-------------|-------------|---------------------------------------------------------------------------------|
| `text_encoders`              | HYVIDTEXTENCODER   | —           | Required    | Text encoder from `DownloadAndLoadHyVideoTextEncoder`                          |
| `prompt`                      | STRING             | ""          | Required    | Text prompt describing the video                                               |
| `force_offload`              | BOOLEAN           | True        | Optional    | Offload text encoder after encoding to save VRAM                               |
| `prompt_template`            | COMBO<sup>2</sup> | "video"     | Optional    | Predefined template for LLM text encoding                                      |
| `custom_prompt_template`     | PROMPT_TEMPLATE    | —           | Optional    | Custom template (used when `prompt_template` is "custom")                      |
| `image`                       | IMAGE              | None        | Optional    | Reference image for image-conditioned generation                               |
| `hyvid_cfg`                  | HYVID_CFG          | —           | Optional    | CFG settings from `HyVideoCFG` node                                            |
| `image_embed_interleave`     | INT                | 2           | Optional    | Interleave frequency for image embeddings in text tokens                       |
| `model_to_offload`           | HYVIDEOMODEL       | None        | Optional    | Offload video model during encoding to save VRAM                               |

**Output:** `HYVIDEMBEDS`  

**Use Cases:**  
- **Image-Prompted Video Generation** (IP2V)  
- Combining **text and image embeddings** for hybrid conditioning  
- Advanced workflows using custom prompt templates  

---

## HyVideoEncodeKeyframes

| **Parameter Name**          | **Type**                        | **Default Value**                              | **Property** | **Description**                                                                 |
| ------------------------------ | ---------------------- | ----------- | ------------- | ------------------------------------------------------- |
| `vae`                          | `VAE`                  | —           | Required      | VAE model to encode images                              |
| `start_image`                  | `IMAGE`                | —           | Required      | First keyframe                                          |
| `end_image`                    | `IMAGE`                | —           | Required      | Last keyframe                                           |
| `num_frames`                   | `INT`                  | `49`        | Required      | Total number of frames between start and end            |
| `enable_vae_tiling`            | `BOOLEAN`              | `True`      | Required      | Whether to use tiled VAE encoding (memory optimization) |
| `temporal_tiling_sample_size`  | `INT`                  | `64`        | Required      | Sample size per temporal chunk                          |
| `spatial_tile_sample_min_size` | `INT`                  | `256`       | Required      | Minimum image tile size                                 |
| `auto_tile_size`               | `BOOLEAN`              | `True`      | Required      | Automatically choose tiling parameters                  |
| `noise_aug_strength`           | `FLOAT`                | `0.0`       | Optional      | Add noise to increase variation (LeapFusion support)    |
| `latent_strength`              | `FLOAT`                | `1.0`       | Optional      | Scale the strength of latent transformation             |
| `latent_dist`                  | `"sample"` or `"mode"` | `sample`    | Optional      | Whether to sample latent distribution or use mode       |

**Output**:  `LATENT` – Interpolated latent tensor between keyframes

**Use Cases**:
* Keyframe-to-video interpolation
* Character pose-to-animation workflows
* Long-frame transitions using minimal image input

---

## HyVideoCFG

| **Parameter Name**          | **Type**                        | **Default Value**                              | **Property** | **Description**                                                                 |
| ------------------- | --------- | ----------- | ------------- | ----------------------------------------------------------------- |
| `cfg`               | `FLOAT`   | `6.0`       | Required      | Base strength of classifier-free guidance                         |
| `curve`             | `STRING`  | `linear`    | Optional      | Interpolation function: linear, ease-in/out, sigmoid, etc.        |
| `start_step`        | `INT`     | `0`         | Optional      | Index to begin applying CFG                                       |
| `end_step`          | `INT`     | `-1`        | Optional      | Index to stop applying CFG (-1 = end)                             |
| `use_time_fraction` | `BOOLEAN` | `False`     | Optional      | Interpret start/end as fractions of total steps (e.g., 0.2 = 20%) |

**Output**:  `CFGARGS` – Classifier-free guidance schedule

**Use Cases**:
* Dynamically adjust prompt strength over time
* Suppress prompt at early or late stages
* Create more subtle or layered semantic influence

---

## HyVideoTextImageEncode

| **Parameter Name**          | **Type**                        | **Default Value**                              | **Property** | **Description**                                                                 |
| ----------------------- | --------------------- | ----------- | ------------- | --------------------------------------------------- |
| `text`                  | `STRING`              | —           | Required      | Prompt text                                         |
| `text_encoder`          | `HYVIDTEXTENCODER`    | —           | Required      | Loaded encoder for processing text                  |
| `image1`                | `IMAGE`               | —           | Required      | First input image                                   |
| `image2`                | `IMAGE`               | —           | Required      | Second input image                                  |
| `vae`                   | `VAE`                 | —           | Required      | VAE model to encode images                          |
| `resolution`            | `INT`                 | `512`       | Optional      | Resize image for encoding                           |
| `is_negative_prompt`    | `BOOLEAN`             | `False`     | Optional      | Whether this embedding is used as a negative prompt |
| `custom_prompt_context` | `CustomPromptContext` | —           | Optional      | Optional text formatting for structured LLM prompts |

**Output**: `HYVIDEMBEDS` – Fused image-text latent embedding

**Use Cases**:
* IP2V: **Image + Prompt → Video** workflows
* Reference-guided animation
* Combines appearance/style of two images with textual behavior/control

---

## HyVideoTextImageEncode

| **Parameter Name**          | **Type**                        | **Default Value**                              | **Property** | **Description**                                                                 |
| ----------------------- | --------------------- | ----------- | ------------- | --------------------------------------------------- |
| `text`                  | `STRING`              | —           | Required      | Prompt text                                         |
| `text_encoder`          | `HYVIDTEXTENCODER`    | —           | Required      | Loaded encoder for processing text                  |
| `image1`                | `IMAGE`               | —           | Required      | First input image                                   |
| `image2`                | `IMAGE`               | —           | Required      | Second input image                                  |
| `vae`                   | `VAE`                 | —           | Required      | VAE model to encode images                          |
| `resolution`            | `INT`                 | `512`       | Optional      | Resize image for encoding                           |
| `is_negative_prompt`    | `BOOLEAN`             | `False`     | Optional      | Whether this embedding is used as a negative prompt |
| `custom_prompt_context` | `CustomPromptContext` | —           | Optional      | Optional text formatting for structured LLM prompts |

**Output**: `HYVIDEMBEDS` – Fused image-text latent embedding

**Use Cases**:
* IP2V: **Image + Prompt → Video** workflows
* Reference-guided animation
* Combines appearance/style of two images with textual behavior/control

---

### HyVideoInverseSampler

| **Parameter Name**          | **Type**                        | **Default Value**                              | **Property** | **Description**                                                                 |
|-----------------------------|---------------------|-------------|-------------|---------------------------------------------------------------------------------|
| `model`                    | HYVIDEOMODEL      | —           | Required    | Preloaded Hunyuan model                                                         |
| `hyvid_embeds`             | HYVIDEMBEDS       | —           | Required    | Empty text embeddings (use with `HyVideoEmptyTextEmbeds`)                       |
| `samples`                  | LATENT            | —           | Required    | Target latent variables (video to invert)                                       |
| `steps`                    | INT               | 30          | Required    | Inverse sampling steps                                                          |
| `gamma`                    | FLOAT             | 0.5         | Required    | Noise interpolation strength (0=full inversion, 1=full forward)                 |
| `gamma_trend`              | COMBO              | "constant"  | Required    | Strength variation trend                                                        |

**Output:** `LATENT` (inverted latents)  

**Use Cases:**  
- Latent space inversion for video editing  
- Frame consistency repair with `HyVideoReSampler`  

---

### HyVideoReSampler

| **Parameter Name**          | **Type**                        | **Default Value**                              | **Property** | **Description**                                                                 |
|-----------------------------|---------------------|-------------|-------------|---------------------------------------------------------------------------------|
| `inversed_latents`         | LATENT            | —           | Required    | Inverted latents (from `HyVideoInverseSampler`)                                |
| `eta_base`                 | FLOAT             | 0.5         | Required    | Base resampling strength                                                       |
| `eta_trend`                | COMBO              | "constant"  | Required    | Temporal strength distribution curve                                           |

**Output:** `LATENT` (optimized latents)  

**Use Cases:**  
- Fix flickering issues in generated videos  
- Enhance temporal consistency for long videos  

---

### HyVideoPromptMixSampler

| **Parameter Name**          | **Type**                        | **Default Value**                              | **Property** | **Description**                                                                 |
|-----------------------------|---------------------|-------------|-------------|---------------------------------------------------------------------------------|
| `hyvid_embeds_2`          | HYVIDEMBEDS       | —           | Required    | Second prompt embeddings                                                       |
| `alpha`                    | FLOAT             | 0.5         | Required    | Prompt mixing sharpness (0-1, higher = abrupt transitions)                     |
| `interpolation_curve`      | FLOAT[]           | —           | Optional    | Per-frame mixing weight curve                                                  |

**Output:** `LATENT` (mixed latents)  

**Use Cases:**  
- Dynamic transitions between prompts (e.g., scene morphing)  
- Multi-style fusion generation  

---

### HyVideoLoraBlockEdit

| **Parameter Name**          | **Type**                        | **Default Value**                              | **Property** | **Description**                                                                 |
| ------------------- | -------- | ----------- | ------------- | ---------------------------------------------- |
| double\_blocks.0–19 | BOOLEAN  | True        | Required      | Toggle injection for each of 20 double blocks. |
| single\_blocks.0–39 | BOOLEAN  | True        | Required      | Toggle injection for each of 40 single blocks. |

**Output:** `SELECTEDBLOCKS`

---

### HyVideoLoraBlockEdit

| **Parameter Name**          | **Type**                        | **Default Value**                              | **Property** | **Description**                                                                 |
| ------------------- | -------- | ----------- | ------------- | ---------------------------------------------- |
| double\_blocks.0–19 | BOOLEAN  | True        | Required      | Toggle injection for each of 20 double blocks. |
| single\_blocks.0–39 | BOOLEAN  | True        | Required      | Toggle injection for each of 40 single blocks. |

**Output**: `SELECTEDBLOCKS`

**Use Cases**:
* Used with `HyVideoLoraSelect` to apply LoRA to specific blocks.
* Enables fine-grained control for style/character injection.
* Used with `HyVideoLoraSelect` to apply LoRA to specific blocks.
* Enables fine-grained control for style/character injection.

---

### HyVideoEnableAVideo (HyVideoEnhanceAVideo)

| **Parameter Name**          | **Type**                        | **Default Value**                              | **Property** | **Description**                                                                 |
|-----------------------------|---------------------|-------------|-------------|---------------------------------------------------------------------------------|
| `weight`                   | FLOAT             | 2.0         | Required    | Enhancement strength (>1.0 boosts details, <1.0 smooths output)                |
| `start_percent`            | FLOAT             | 0.0         | Required    | Starting step percentage for enhancement                                        |
| `end_percent`              | FLOAT             | 1.0         | Required    | Ending step percentage for enhancement                                          |

**Output**:`FETAARGS` 

**Use Cases**:  
- Repair low-quality generations  
- Enhance video texture details  

---

### HyVideoSTG

| **Parameter Name**          | **Type**                        | **Default Value**                              | **Property** | **Description**                                                                 |
| ------------------- | -------- | ----------- | ------------- | ---------------------------------------------------------------- |
| stg\_mode           | COMBO    | STG-A       | Required      | Mode of spatio-temporal guidance: `STG-A` or `STG-R`.            |
| stg\_block\_idx     | INT      | 0           | Required      | Block index where guidance is applied (-1 to 39).                |
| stg\_scale          | FLOAT    | 1.0         | Required      | Strength of STG effect (recommended ≤ 2.0).                      |
| stg\_start\_percent | FLOAT    | 0.0         | Required      | When to start applying guidance (as percent of denoising steps). |
| stg\_end\_percent   | FLOAT    | 1.0         | Required      | When to end applying guidance (as percent of denoising steps).   |

**Output**: `STGARGS`

**Use Cases**:
* Applied to `HyVideoSampler` to preserve temporal consistency.
* Works well with keyframes and long motion scenes.

---

### HunyuanVideoFresca

| **Parameter Name**          | **Type**                        | **Default Value**                              | **Property** | **Description**                                                                 |
| -------------------- | -------- | ----------- | ------------- | ------------------------------------------------------------- |
| fresca\_scale\_low   | FLOAT    | `1.0`       | Required      | Frequency modulation scaling for low-end frequency bands      |
| fresca\_scale\_high  | FLOAT    | `1.25`      | Required      | Frequency modulation scaling for high-end bands               |
| fresca\_freq\_cutoff | INT      | `20`        | Required      | Frequency cutoff threshold for separating low/high modulation |

**Output**: `FRESCA_ARGS`

**Use Cases**:
* Apply frequency-based style transfer
* Animate textures or features using spectral modulation

---

### HyVideoLatentPreview

| **Parameter Name**          | **Type**                        | **Default Value**                              | **Property** | **Description**                                                                 |
|-----------------------------|---------------------|-------------|-------------|---------------------------------------------------------------------------------|
| `min_val`                  | FLOAT             | -0.15       | Required    | Minimum value mapping for latent visualization                                  |
| `max_val`                  | FLOAT             | 0.15        | Required    | Maximum value mapping for latent visualization                                  |
| `rgb_bias`                 | FLOAT[3]          | [0,0,0]     | Required    | RGB channel bias adjustment                                                     |

**Output:** `IMAGE` (preview), `STRING` (color mapping formula)  

**Use Cases:**  
- Latent space debugging and visualization  
- Quick content check without full decoding  

---

### HyVideoLatentPreview

| **Parameter Name**          | **Type**                        | **Default Value**                              | **Property** | **Description**                                                                 |
|-----------------------------|---------------------|-------------|-------------|---------------------------------------------------------------------------------|
| `min_val`                  | FLOAT             | -0.15       | Required    | Minimum value mapping for latent visualization                                  |
| `max_val`                  | FLOAT             | 0.15        | Required    | Maximum value mapping for latent visualization                                  |
| `rgb_bias`                 | FLOAT[3]          | [0,0,0]     | Required    | RGB channel bias adjustment                                                     |

**Output**: `IMAGE` (preview), `STRING` (color mapping formula)  

**Use Cases**:  
- Latent space debugging and visualization  
- Quick content check without full decoding  

---

### HyVideoCustomPromptTemplate

| **Parameter Name**          | **Type**                        | **Default Value**                              | **Property** | **Description**                                                                 |
|-----------------------------|---------------------|-------------|-------------|---------------------------------------------------------------------------------|
| `custom_prompt_template`  | STRING             | —           | Required    | Custom template (must include `{}` placeholder)                                |
| `crop_start`               | INT                | 0           | Required    | System prompt truncation position                                              |

**Output**: `PROMPT_TEMPLATE`  

**Use Cases**:
- Prompt engineering for non-standard language models  
- Unified template management for multi-task workflows  

---

### HyVideoSTG

| **Parameter Name**          | **Type**                        | **Default Value**                              | **Property** | **Description**                                                                 |
| ------------------- | -------- | ----------- | ------------- | ---------------------------------------------------------------- |
| stg\_mode           | COMBO    | STG-A       | Required      | Mode of spatio-temporal guidance: `STG-A` or `STG-R`.            |
| stg\_block\_idx     | INT      | 0           | Required      | Block index where guidance is applied (-1 to 39).                |
| stg\_scale          | FLOAT    | 1.0         | Required      | Strength of STG effect (recommended ≤ 2.0).                      |
| stg\_start\_percent | FLOAT    | 0.0         | Required      | When to start applying guidance (as percent of denoising steps). |
| stg\_end\_percent   | FLOAT    | 1.0         | Required      | When to end applying guidance (as percent of denoising steps).   |

**Output**: `STGARGS`

**Use Cases**:
* Applied to `HyVideoSampler` to preserve temporal consistency.
* Works well with keyframes and long motion scenes.

---

### HunyuanVideoFresca

| **Parameter Name**          | **Type**                        | **Default Value**                              | **Property** | **Description**                                                                 |
| -------------------- | -------- | ----------- | ------------- | ------------------------------------------------------------- |
| fresca\_scale\_low   | FLOAT    | `1.0`       | Required      | Frequency modulation scaling for low-end frequency bands      |
| fresca\_scale\_high  | FLOAT    | `1.25`      | Required      | Frequency modulation scaling for high-end bands               |
| fresca\_freq\_cutoff | INT      | `20`        | Required      | Frequency cutoff threshold for separating low/high modulation |

**Output**: `FRESCA_ARGS`

**Use Cases**:
* Apply frequency-based style transfer
* Animate textures or features using spectral modulation

---

### HunyuanVideoSLG

| **Parameter Name**          | **Type**                        | **Default Value**                              | **Property** | **Description**                                                                 |
|-----------------------------|---------------------|-------------|-------------|---------------------------------------------------------------------------------|
| `single_blocks`            | STRING             | "20"        | Required    | Single-block indices to skip uncond computation (comma-separated)              |
| `start_percent`            | FLOAT             | 0.4         | Required    | SLG activation start step percentage                                            |
| `end_percent`              | FLOAT             | 0.8         | Required    | SLG activation end step percentage                                              |

**Output**: `SLGARGS`  

**Use Cases**:  
- Reduce motion blur in high-dynamic scenes  
- Speed up generation (sacrifices some diversity)  

---

### HyVideoLoopArgs

| **Parameter Name**          | **Type**                        | **Default Value**                              | **Property** | **Description**                                                                 |
| -------------- | -------- | ----------- | ------------- | --------------------------------------------- |
| shift\_skip    | INT      | `6`         | Required      | Latent shift step count to skip for each loop |
| start\_percent | FLOAT    | `0.0`       | Required      | Percent of generation where loop starts       |
| end\_percent   | FLOAT    | `1.0`       | Required      | Percent of generation where loop ends         |

**Output**: `LOOPARGS`

**Use Cases**:
* Enable seamless video loop effect
* Supports latent-shift interpolation inspired by Möbius method

---

### HyVideoContextOptions

| **Parameter Name**          | **Type**                        | **Default Value**                              | **Property** | **Description**                                                                 |
| ----------------- | -------- | ------------------ | ------------- | -------------------------------------------------------------- |
| context\_schedule | COMBO    | `uniform_standard` | Required      | Strategy for context window scheduling                         |
| context\_frames   | INT      | `65`               | Required      | Number of pixel frames per context window (4 latent = 1 pixel) |
| context\_stride   | INT      | `4`                | Required      | Step between each context window                               |
| context\_overlap  | INT      | `4`                | Required      | Overlap between context windows                                |
| freenoise         | BOOLEAN  | `True`             | Required      | Whether to shuffle noise between context chunks                |

**Output**: `HYVIDCONTEXT`

**Use Cases**:
* Split long video sequences into context-aware chunks
* Optimize memory use while maintaining coherence across segments

----

### 🔄 Common Workflow Combinations

| Workflow Name             | Node Combination                                                            | Purpose                                  |
| ------------------------- | --------------------------------------------------------------------------- | ---------------------------------------- |
| hyvideo_dashtoon_keyframe_example_01(First and last frame video generation)  | HyVideoTorchCompileSettings → HyVideoVAELoader → Reroute → HyVideoEncodeKeyframes & HyVideoDecode → HyVideoSampler → HyVideoDecode → VHS_VideoCombine                   | Keyframe image-guided video generation                 |
| hunhyuan_rf_inversion_testing_01(video-to-video generation)        | VHS_LoadVideo → ImageResizeKJ → Set_InputVideo → GetImageSizeAndCount → HyVideoEncode → HyVideoInverseSampler → HyVideoReSampler → HyVideoDecode → ImageConcatMulti → VHS_VideoCombine                          | reverse inference + resampling test           |
| hyvideo_ip2v_experimental_dango(image-to-video generation)  | LoadImage → HyVideoGetClosestBucketSize → ImageScale → GetImageSizeAndCount → HyVideoEncode → HyVideoI2VEncode → HyVideoSampler → HyVideoDecode → GetImageSizeAndCount → ImageConcatMulti → VHS_VideoCombine                         | image-driven video generation    |
| text-to-video generation(hyvideo_t2v_example_01)     | DownloadAndLoadHyVideoTextEncoder → HyVideoTextEncode → HyVideoSampler → HyVideoDecode → VHS_VideoCombine | text-driven video generation |