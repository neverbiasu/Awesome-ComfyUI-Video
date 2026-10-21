# 📹 ComfyUI-CogvideoXWrapper

## 🔍 Module Overview

| Item     | Details                                                                    |
| -------- | -------------------------------------------------------------------------- |
| 📌 **Author** | [@Kkijai](https://github.com/kijai)                                        |
| 📅 **Version** | 1.2.0+                                                                     |
| 🏷️ **Category** | Open-source Model Extension                                                |
| 🔗 **Repository** | [GitHub Link](https://github.com/kijai/ComfyUI-CogVideoXWrapper)       |

## 📝 Functionality Overview

Developed by kijai, this open-source extension seamlessly integrates the cutting-edge CogVideoX large-scale text-to-video model into the node-based ComfyUI interface. It features a complete set of custom nodes for model downloading, input encoding, sampling inference, and result decoding, enabling diverse functionalities such as "Text-to-Video" (T2V), "Image-to-Video" (I2V), and "Video Style Transfer".

## 📊 Node Overview Table

| Node Name                          | Type      | Main Function                                                               | Complexity | Common Use Cases                                           |
| ---------------------------------- | --------- | --------------------------------------------------------------------------- | ---------- | ---------------------------------------------------------- |
| **(Down)load CogVideo Model**      | I/O       | Downloads and loads CogVideo standard format models from HuggingFace.       | ★☆☆☆☆      | Acquire models before first use.                           |
| **(Down)load CogVideo GGUF Model** | I/O       | Downloads and loads CogVideo models in GGUF format.                         | ★☆☆☆☆      | Load models in GGUF-only environments.                     |
| **(Down)load Tora Model**          | I/O       | Downloads and loads CogVideoX Tora-optimized models from HuggingFace.       | ★☆☆☆☆      | Accelerate inference using Alibaba's Tora optimized version.|
| **CogVideoX Model Loader**         | I/O       | Loads a CogVideoX model from a specified path or cache.                     | ★☆☆☆☆      | Custom model management and reuse.                         |
| **CogVideoX VAE Loader**           | I/O       | Loads or switches VAE decoders.                                             | ★☆☆☆☆      | Switch VAEs between different precisions/formats.          |
| **(Down)load CogVideo ControlNet** | I/O       | Downloads and loads CogVideo ControlNet conditional guidance modules.       | ★☆☆☆☆      | Add ControlNet guidance to video generation; pose/structure-driven generation.|
| **CogVideo ControlNet**            | I/O       | Applies ControlNet conditional networks, injecting specific guidance (e.g., pose, depth, bounding box) into the generation process. | ★★☆☆☆      | Pose-driven, structure-driven, style matching, and other external conditional guidance.|
| **CogVideo TextEncode**            | Process   | Encodes a single text prompt, outputting text latent vectors.               | ★★☆☆☆      | Pre-encoding for Text-to-Video pipelines.                  |
| **CogVideo TextEncode Combine**    | Process   | Combines multiple text latent vectors, providing a unified input for subsequent sampling. | ★★☆☆☆      | Combine multiple prompts for complex scene generation.     |
| **CogVideo ImageEncode**           | I/O       | Encodes a static image into a spatiotemporal latent vector usable for video. | ★★★☆☆      | Image encoding in I2V workflows.                           |
| **CogVideo ImageEncode FunInP**    | I/O       | Encodes images for Fun-InP (unofficial I2V) models.                         | ★★★☆☆      | Use CogVideoX-Fun for Image-to-Video.                      |
| **CogVideo Sampler**               | Process   | Performs diffusion sampling for video frames based on text/image latent vectors. | ★★★★☆      | Core sampling for T2V or I2V.                              |
| **CogVideo Decode**                | I/O       | Decodes latent vectors back into an image sequence and outputs as a video.  | ★★☆☆☆      | Generate final video output after sampling.                |
| **CogVideoXFun ResizeToClosestBucket** | Process | Automatically adjusts latent vectors to the nearest sampling "bucket" size. | ★☆☆☆☆      | Ensure resolution compatibility with Fun models.           |
| **CogVideo FasterCache**           | Utility   | Enables FasterCache optimization, trading a small amount of VRAM for higher inference speed. | ★★☆☆☆      | Speed improvement for long videos or high-resolution scenes.|
| **CogVideo TorchCompileSettings**  | Utility   | Configures `torch.compile` optimization options.                            | ★★☆☆☆      | Use Triton/SageAttention combinations for compilation acceleration.|
| **CogVideo Context Options**       | Utility   | Sets context window size and FreeNoise noise shuffling strategy.            | ★★☆☆☆      | Long sequence context management for Vid2Vid or Pose2Vid.  |
| **CogVideo LatentPreview**         | Utility   | Previews the effect of intermediate latent vectors in the node panel.       | ★☆☆☆☆      | Debug or visualize latent space generation effects.        |
| **CogVideo Enhance-A-Video**       | Process   | Performs post-processing on output video for brightening, denoising, or stylization. | ★★★☆☆      | Post-generation enhancement, e.g., color correction, flicker removal.|
| **CogVideo LoraSelect**            | Advanced  | Dynamically inserts/switches LoRA weights by name or tag within the pipeline. | ★★☆☆☆      | Quickly experiment with different LoRA effects.            |
| **CogVideo LoraSelect Comfy**      | Advanced  | Seamless integration with ComfyUI's native LoRA management system.         | ★★★☆☆      | Collaborate with other ComfyUI LoRA nodes.                 |
| **CogVideo TransformerEdit**       | Advanced  | Prunes specified Transformer Blocks, removing unnecessary layers to reduce VRAM usage and improve inference efficiency. | ★★★★☆      | Model lightweighting; experimental layer comparison; accelerated generation in resource-constrained environments.|
| **Tora Encode Trajectory**         | Process   | Uses Tora's trajectory encoder to convert user-drawn motion paths into spatiotemporal motion patch latent vectors. | ★★★★☆      | Precise motion control in I2V; animation creation.         |

---

## 📑 Core Node Details

### (Down)load CogVideo Model

| Parameter Name                  | Type                       | Default Value   | Parameter Nature | Function Description                                                                                  |
| :------------------------------ | :------------------------- | :-------------- | :------- | :---------------------------------------------------------------------------------------------------- |
| `model`                         | STRING                     | —               | Required | The CogVideoX model identifier to download and load, e.g., `THUDM/CogVideoX-2b`, `kijai/CogVideoX-5b-Tora`. |
| `block_edit`                    | TRANSFORMERBLOCKS          | —               | Optional | A comma-separated list of Transformer Block indices. Specified layers will be pruned or replaced during loading to adjust model capacity and performance. |
| `lora`                          | COGLORA                    | —               | Optional | Specifies LoRA weights (path or name) to be automatically applied during loading, for style fine-tuning or effect enhancement. |
| `compile_args`                  | COMPILEARGS                | —               | Optional | Additional arguments passed to `torch.compile` or `diffusers` compilation interfaces, to control backend optimization behavior. |
| `model.precision`               | ENUM (`fp16`,`fp32`,`bf16`) | `fp16`          | Optional | Specifies the weight precision when loading the model. Lower precision saves VRAM but may slightly reduce quality. |
| `quantization`                  | ENUM                       | `disabled`      | Optional | Quantization backend options, such as `fp8_e4m3fn`, `torchao_int8dq`, to help save memory on large models. |
| `enable_sequential_cpu_offload` | BOOLEAN                    | `False`         | Optional | If enabled, sequentially offloads model submodules between CPU/GPU to significantly reduce peak VRAM usage. |
| `attention_mode`                | ENUM                       | `sdpa`          | Optional | Selects the attention implementation, e.g., `fused_sdpa`, `sageattn_qk_int8_pv_fp16_cuda`, `comfy`, to optimize speed/memory. |
| `load_device`                   | ENUM (`main_device`,`offload_device`) | `main_device`   | Optional | Controls the initial loading location of model components: `main_device` (GPU) or `offload_device` (CPU). |

**Outputs**  
- `COGVIDEOMODEL`: The `CogVideoXPipeline` object, loaded and configured according to the parameters.  
- `VAE`: The VAE decoder module that comes with the model.  

**Use Cases**  
- Automatically optimizes model loading for different hardware environments (multi-GPU / CPU + GPU).  
- Balances speed, VRAM, and generation quality by combining LoRA, quantization, pruning, etc.  
- Quickly switches between various CogVideoX series models and versions for comparative testing.  

---

### (Down)load CogVideo GGUF Model

| Parameter Name                  | Type         | Default Value | Parameter Nature | Function Description                                                                                 |
| :------------------------------ | :----------- | :------------ | :------- | :--------------------------------------------------------------------------------------------------- |
| `model`                         | STRING       | —             | Required | The name of the CogVideo GGUF model to download and load.                                            |
| `vae_precision`                 | STRING       | `fp16`        | Optional | Precision for the VAE component, selectable from `bf16`, `fp16`, `fp32`.                             |
| `fp8_fastmode`                  | BOOLEAN      | `False`       | Optional | Whether to enable FP8 fast mode, which improves performance but may slightly reduce accuracy.          |
| `load_device`                   | STRING       | `cuda`        | Optional | Device for model loading, selectable from `cpu` or `cuda`.                                           |
| `enable_sequential_cpu_offload` | BOOLEAN      | `False`       | Optional | Whether to enable sequential CPU offloading, selectively offloading model components to CPU to save GPU VRAM. |
| `attention_model`               | STRING       | `sdpa`        | Optional | Attention implementation method, selectable from `sdpa` (standard) or `sageattn` (SageAttention acceleration, Linux only). |
| `block_edit`                    | LIST[int]    | —             | Optional | Specifies a list of Transformer Block indices to modify or remove for model structure fine-tuning.    |

**Outputs**  
- `model`: The loaded and configured CogVideo GGUF model object.  
- `vae`: The VAE decoder module with corresponding precision.  

**Use Cases**  
- In low-VRAM or mobile environments, loads quantized GGUF models with matching VAE precision to balance performance and resource consumption.  

---

### (Down)load Tora Model

| Parameter Name | Type   | Default Value               | Parameter Nature | Function Description                                                      |
| :------------- | :----- | :-------------------------- | :------- | :------------------------------------------------------------------------ |
| `model_name`   | STRING | `"kijai/CogVideoX-5b-Tora"` | Optional | The HuggingFace repository name of the Tora-optimized model to download.  |

**Outputs**  
- `TORAMODEL`: The downloaded and instantiated Tora-optimized model object, ready for subsequent sampling and decoding nodes.  

**Use Cases**  
- When deploying or testing Tora-optimized CogVideoX models, this node allows one-click download and loading into VRAM, supporting both standard T2V and dedicated I2V versions.  

---

### CogVideoX Model Loader

| Parameter Name                  | Type                        | Default Value   | Parameter Nature | Function Description                                                                                   |
| :------------------------------ | :-------------------------- | :-------------- | :------- | :----------------------------------------------------------------------------------------------------- |
| `model`                         | MODEL                       | —               | Required | The CogVideoX model object or identifier (local path or cached name) to load.                          |
| `base_precision`                | ENUM(`fp16`,`fp32`,`bf16`)  | `fp16`          | Optional | The base precision for model weights, affecting VRAM usage and numerical range.                        |
| `quantization`                  | ENUM(...)                   | `disabled`      | Optional | Quantization mode, supporting various FP8/INT8/FP6 schemes, for further VRAM reduction and inference acceleration. |
| `enable_sequential_cpu_offload` | BOOLEAN                     | `False`         | Optional | Whether to enable sequential CPU offloading, transferring some submodule weights from GPU to CPU to save VRAM. |
| `block_edit`                    | TRANSFORMERBLOCKS           | —               | Optional | Specifies a list of Transformer Blocks to prune or reorder (generated by `CogVideo TransformerEdit`).  |
| `lora`                          | COGLORA                     | —               | Optional | LoRA weight configuration to apply (provided by `CogVideo LoraSelect`/`Comfy` nodes).                |
| `compile_args`                  | COMPILEARGS                 | —               | Optional | Arguments passed to `torch.compile` for model compilation optimization (e.g., mode, backend, autotune). |
| `attention_mode`                | ENUM(...)                   | `sdpa`          | Optional | Attention computation mode, various SDPA/Sage/Fused and Comfy custom schemes, affecting speed and precision. |

**Outputs**  
- `COGVIDEOMODEL`: The loaded and further manipulable CogVideoX model instance.  

**Use Cases**  
- Flexibly loads CogVideoX models in different formats or precisions at the start of a ComfyUI pipeline.  
- Customizes model performance and resource usage by combining with LoRA, Transformer pruning, `torch.compile`, and other nodes.  

---

### CogVideoX VAE Loader

| Parameter Name | Type   | Default Value            | Parameter Nature | Function Description                                                      |
| :------------- | :----- | :----------------------- | :------- | :------------------------------------------------------------------------ |
| `model_name`   | STRING | `"THUDM/CogVideoX-2b"`   | Optional | The CogVideoX model name or local path to load, supporting Diffusers repository format. |
| `precision`    | STRING | `"fp16"`                 | Optional | Specifies the VAE data type to load, selectable from `"fp16"`, `"fp32"`, or `"bf16"`. |

**Outputs**  
- `VAE`: The loaded 3D VAE instance (`AutoencoderKLCogVideoX`), used to restore latent vectors to video frames during decoding.  

**Use Cases**  
- Switches VAE precision based on VRAM and performance requirements (e.g., `fp16` for low VRAM, `fp32` or `bf16` for high fidelity).  
- Loads separately from the main model for easy reuse or replacement of the VAE decoder to test different decoding effects.  

---

### (Down)load CogVideo ControlNet

| Parameter Name | Type          | Default Value | Parameter Nature | Function Description                                                                                       |
| :------------- | :------------ | :------------ | :------- | :--------------------------------------------------------------------------------------------------------- |
| `model`        | COMBO[STRING] | —             | Required | Selects the CogVideoX ControlNet model name to download and load from a dropdown list.                     |

**Outputs**  
- `COGVIDECONTROLNETMODEL`: The downloaded and loaded ControlNet model object, usable as input for subsequent Apply ControlNet nodes.  

**Use Cases**  
- Injects conditional guidance such as pose (HED) or edges (Canny) into the video generation process for fine control over the generated content.  

---

### CogVideo TextEncode

| Parameter Name | Type    | Default Value | Parameter Nature | Function Description                                            |
| :------------- | :------ | :------------ | :------- | :-------------------------------------------------------------- |
| `clip`         | CLIP    | —             | Required | Provides a CLIP model instance for text encoding, converting prompt text into embedding vectors.  |
| `prompt`       | STRING  | `""`          | Required | The input text prompt, used to guide the semantic content of the video generation.              |
| `strength`     | FLOAT   | `1.0`         | Optional | Controls the strength of the text embedding; higher values mean a more pronounced prompt influence. |
| `force_offload`| BOOLEAN | `False`       | Optional | Whether to force offload the model to CPU to save GPU VRAM.     |

**Outputs**  
- `CONDITIONING`: Text conditioning embedding for subsequent sampling.  
- `CLIP`: Returns the CLIP model instance for reuse.  

**Use Cases**  
- Converts natural language prompts into model-recognizable conditional vectors for text-to-video or text-to-image workflows.  

---

### CogVideo Decode

| Parameter Name              | Type    | Default Value | Parameter Nature | Function Description                                                     |
| :-------------------------- | :------ | :------------ | :------- | :----------------------------------------------------------------------- |
| `vae`                       | VAE     | —             | Required | The VAE model used for decoding.                                         |
| `samples`                   | LATENT  | —             | Required | Input latent vectors, from the output of a sampling node.                |
| `enable_vae_tiling`         | BOOLEAN | —             | Optional | Whether to enable VAE tiling for decoding, processing in blocks to reduce VRAM pressure. |
| `tile_sample_min_height`    | INT     | —             | Optional | Minimum height of each tile during tiled decoding; tiles smaller than this will be merged. |
| `tile_sample_min_width`     | INT     | —             | Optional | Minimum width of each tile during tiled decoding.                        |
| `tile_overlap_factor_height`| FLOAT   | —             | Optional | Overlap ratio of tiles in the height direction, for smoothing boundaries. |
| `tile_overlap_factor_width` | FLOAT   | —             | Optional | Overlap ratio of tiles in the width direction, for smoothing boundaries. |
| `auto_tile_size`            | BOOLEAN | —             | Optional | Whether to automatically calculate optimal tile parameters based on input size. |

**Outputs**  
- `IMAGE`: Decoded image sequence, ready for video composition or further processing.  

**Use Cases**  
- Decodes latent representations from the CogVideoX Sampler into visual video frames using the VAE,  
  also capable of processing high-resolution or long-duration videos in tiled mode to save VRAM and reduce decoding errors.  

---

### CogVideo TextEncode Combine

| Parameter Name         | Type   | Default Value        | Parameter Nature | Function Description                                                                                   |
| :--------------------- | :----- | :------------------- | :------- | :----------------------------------------------------------------------------------------------------- |
| `conditioning_1`       | TENSOR | —                    | Required | The first text latent vector input, typically from `CogVideo TextEncode` or `DualTextEncode` nodes.      |
| `conditioning_2`       | TENSOR | —                    | Required | The second text latent vector input, with the same shape as `conditioning_1`, for combination.           |
| `combination_mode`     | STRING | `"weighted_average"` | Optional | Combination mode, selectable: <br>• `"average"`: Simple average<br>• `"weighted_average"`: Weighted average<br>• `"concatenate"`: Concatenate along the last dimension |
| `weighted_average_ratio`| FLOAT  | `0.5`                | Optional | Effective when `combination_mode="weighted_average"`, range 0.0–1.0, controls the weight ratio of the two inputs.    |

**Outputs**  
- `conditioning`: The combined text latent vector (TENSOR), ready for input to a sampler node (e.g., `CogVideo Sampler`) for video generation.  

**Use Cases**  
- Combines multiple prompts (e.g., positive and negative prompts, or different thematic prompts) into a single guiding vector, offering flexible control over the style and details of generated content.  
- Preserves full features of each input with `"concatenate"` mode, or achieves smooth transitions and balance with weighted average.  

---

### CogVideo Sampler

| Parameter Name          | Type              | Default Value     | Parameter Nature | Function Description                                                                                   |
| :---------------------- | :---------------- | :---------------- | :------- | :----------------------------------------------------------------------------------------------------- |
| `model`                 | MODEL             | —                 | Required | The loaded CogVideoX model instance, used for actual sampling.                                           |
| `positive`              | TENSOR            | —                 | Required | Positive prompt latent vector, guiding the sampling process.                                             |
| `negative`              | TENSOR            | —                 | Optional | Negative prompt latent vector, used for classifier-free guidance.                                        |
| `samples`               | INT               | 1                 | Optional | Number of video samples to generate per call.                                                            |
| `images_cond_latent`    | LATENT            | —                 | Optional | Image condition encoded latent vector (in I2V mode), output from `CogVideo ImageEncode` node.          |
| `context_options`       | DICT              | —                 | Optional | Context configuration generated by the `CogVideo Context Options` node.                                  |
| `controlnet`            | LIST[MODULE, FLOAT]| []                | Optional | Multiple ControlNet modules and their strengths (loaded by `(Down)load CogVideo ControlNet` node).       |
| `tora_trajectory`       | TENSOR            | —                 | Optional | `Tora Encode Trajectory` node output of spatiotemporal motion patch latent vector.                     |
| `fastercache`           | DICT              | —                 | Optional | `CogVideo FasterCache` node generated cache optimization configuration.                                |
| `feta_args`             | DICT              | —                 | Optional | Additional parameters passed to the underlying sampler (e.g., fp8 mode).                                 |
| `num_frames`            | INT               | 16                | Optional | Total number of video frames to generate.                                                                |
| `steps`                 | INT               | 50                | Optional | Diffusion sampling steps.                                                                                |
| `cfg`                   | FLOAT             | 7.5               | Optional | Classifier-free guidance strength.                                                                       |
| `seed`                  | INT               | 0                 | Optional | Random seed.                                                                                           |
| `control_after_generate`| BOOLEAN           | False             | Optional | Whether to apply ControlNet conditions after latent sampling is complete.                                |
| `scheduler`             | SCHEDULER         | PNDMScheduler     | Optional | Noise scheduler instance.                                                                                |
| `denoise_strength`      | FLOAT             | 1.0               | Optional | Controls denoising strength in Vid2Vid/style transfer workflows.                                         |

**Outputs**  
- `samples`: Contains the generated video latent tensor or decoded video frames, specific to the subsequent decode node configuration.  

**Use Cases**  
- Core Text-to-Video, Image-to-Video, Video-to-Video diffusion sampling node, generating high-quality video samples with rich conditioning, context, and optimization options.  

---

### CogVideo ImageEncode

| Parameter Name        | Type    | Default Value | Parameter Nature | Function Description                                                                                  |
| :-------------------- | :------ | :------------ | :------- | :---------------------------------------------------------------------------------------------------- |
| `vae`                 | VAE     | —             | Required | Specifies the Variational Autoencoder used for encoding, to map images to spatiotemporal latent space.        |
| `start_image`         | IMAGE   | —             | Required | The starting frame image, as the first keyframe input for video generation.                           |
| `end_image`           | IMAGE   | —             | Optional | The ending frame image, used to interpolate intermediate frames between `start_image` and `end_image`. |
| `enable_tiling`       | BOOLEAN | False         | Optional | Whether to enable tiling for encoding, processing in blocks to reduce VRAM usage, suitable for ultra-high-resolution images. |
| `noise_aug_strength`  | FLOAT   | 0.0           | Optional | The strength of noise to add to the input image latent vectors; higher values mean more noticeable perturbation, usable for adding dynamic effects or jitter. |
| `strength`            | FLOAT   | 1.0           | Optional | Controls the preservation ratio of original image features in the latent representation; smaller values lean towards random noise, larger values preserve more original image details. |
| `start_percent`       | FLOAT   | 0.0           | Optional | Proportion of the start image in the blend during interpolation (0.0–1.0), controlling the starting weight of the transition from `start_image` to `end_image`. |
| `end_percent`         | FLOAT   | 1.0           | Optional | Proportion of the end image in the blend during interpolation (0.0–1.0), controlling the weight at the end of the transition. |

**Outputs**  
- `LATENT`: Spatiotemporal latent representation with shape `[batch, num_frames, channels, height, width]`, ready for direct input to `CogVideo Sampler` or other downstream nodes.  

**Use Cases**  
- **Image-to-Video (I2V)**: Generates coherent video sequences by interpolating static images or between two images.  
- **Animation Production**: Combines `start_image` and `end_image` with adjustable `start_percent`/`end_percent` to achieve frame-by-frame transition animations.  
- **Large-Scale Encoding**: Enables `enable_tiling` when processing high-resolution images to reduce peak VRAM.    

---

### CogVideo ImageEncode FunInP

| Parameter Name       | Type    | Default Value | Parameter Nature | Function Description                                                                                       |
| :------------------- | :------ | :------------ | :------- | :--------------------------------------------------------------------------------------------------------- |
| `vae`                | VAE     | —             | Required | The VAE decoder instance used for encoding.                                                                |
| `start_image`        | IMAGE   | —             | Required | The starting frame image, used as the first frame input for spatiotemporal encoding.                       |
| `end_image`          | IMAGE   | —             | Required | The ending frame image, used as the last frame input for spatiotemporal encoding.                          |
| `num_frames`         | INT     | 16            | Optional | Number of intermediate frames to generate.                                                                 |
| `enable_tiling`      | BOOLEAN | False         | Optional | Whether to perform tiled encoding on input images to support ultra-high-resolution images.                 |
| `noise_aug_strength` | FLOAT   | 0.0           | Optional | Strength of noise enhancement added to images during encoding, for increasing randomness or covering imperfections. |

**Outputs**  
- `LATENT`: Spatiotemporal latent vector with shape `[batch, num_frames, ...]`, ready for direct input to the Sampler node for diffusion sampling.  

**Use Cases**  
- Image-to-Video (I2V) workflow: Automatically encodes two static images and intermediate frames into video latent representations for text-guided or text-free scenarios, suitable for character movement, object translation, and similar animation effects.  
- Ultra-high-resolution images: Enabling tiling (`enable_tiling=True`) allows encoding large images in blocks, preventing VRAM overflow, while `noise_aug_strength` controls noise consistency per block.  

---

### Tora Encode Trajectory

| Parameter Name   | Type        | Default Value | Parameter Nature | Function Description                                                         |
| :--------------- | :---------- | :------------ | :------- | :--------------------------------------------------------------------------- |
| `tora_model`     | TORAMODEL   | —             | Required | The loaded Tora model, used to generate spatiotemporal motion features.      |
| `vae`            | VAE         | —             | Required | The VAE decoder module, used to map motion patches to latent space.          |
| `coordinates`    | STRING      | —             | Required | User-defined motion trajectory coordinates (JSON/CSV string), describing the motion path. |
| `width`          | INT         | —             | Optional | Spatial width of the trajectory patch (pixels), should match original image width. |
| `height`         | INT         | —             | Optional | Spatial height of the trajectory patch (pixels), should match original image height. |
| `num_frames`     | INT         | —             | Optional | Number of trajectory patch frames to generate.                               |
| `strength`       | FLOAT       | —             | Optional | Trajectory encoding strength, controlling the influence ratio of motion features. |
| `start_percent`  | FLOAT       | —             | Optional | Start injection percentage during the sampling process (0.0–1.0), determining when to start applying the trajectory. |
| `end_percent`    | FLOAT       | —             | Optional | End injection percentage during the sampling process (0.0–1.0), determining when to stop applying the trajectory. |
| `enable_tiling`  | BOOLEAN     | —             | Optional | Whether to enable tiled processing, generating motion patches in batches to reduce VRAM usage. |

**Outputs**  
- `TORAFEATURES`: Encoded spatiotemporal motion feature tensor, ready for direct input to sampling nodes.  
- `IMAGE`: Motion trajectory visualization, for debugging and previewing trajectory distribution.  

**Use Cases**  
- In I2V workflows, converts user-drawn paths into motion patches understood by the Tora model, precisely controlling the main subject's motion trajectory in the generated video.  

---

### CogVideo ControlNet

| Parameter Name        | Type    | Default Value | Parameter Nature | Function Description                                                                 |
| :-------------------- | :------ | :------------ | :------- | :----------------------------------------------------------------------------------- |
| `control_image`       | IMAGE   | —             | Required | Conditional image/video frame for control, such as Canny, HED, Pose, or depth maps.  |
| `controlnet_strength` | FLOAT   | 1.0           | Optional | Control signal strength, determining the influence ratio of the ControlNet condition on the final sampling (0.0–2.0). |
| `start_percent`       | FLOAT   | 0.0           | Optional | Relative position in total sampling steps where control influence begins (0.0–1.0), e.g., 0.2 means after 20% of steps. |
| `end_percent`         | FLOAT   | 1.0           | Optional | Relative position in total sampling steps where control influence ends (0.0–1.0), e.g., 0.8 means after 80% of steps. |

**Outputs**  
- `controlnet_states`: Processed conditional latent sequence, ready for direct input to CogVideo Sampler for fused sampling.  

**Use Cases**  
- Introduces structural, pose, depth, or other visual information into Text-to-Video, Image-to-Video, or Video-to-Video workflows, allowing fine-grained control over video content layout and motion.  

---

### CogVideoXFun ResizeToClosestBucket

| Parameter Name      | Type    | Default Value   | Parameter Nature | Function Description                                                                                           |
| :------------------ | :------ | :------------ | :------- | :------------------------------------------------------------------------------------------------------------- |
| `images`            | IMAGE   | —             | Required | Input image or latent vector sequence (frames), to be resized to comply with model "bucket" requirements.       |
| `base_resolution`   | INT     | —             | Required | The minimum resolution bucket compatible with the model, e.g., 256, 384, 512; output will align to this and its multiples. |
| `upscale_method`    | STRING  | `nearest-exact` | Optional | Upscaling method used during resolution adjustment: `nearest-exact`, `bilinear`, `area`, `bicubic`, or `lanczos`. |
| `crop`              | STRING  | `center`        | Optional | Cropping mode when original resolution is higher than target bucket: `disabled` (no cropping) or `center` (center crop). |

**Outputs**  
- `IMAGE`: Image/latent vector sequence resized to the nearest bucket size.  
- `INT`: Adjusted image width (pixels).  
- `INT`: Adjusted image height (pixels).  

**Use Cases**  
- Automatically aligns images or latent sequences to the nearest "bucket" resolution before using CogVideoX-Fun or other models with strict input resolution requirements, preventing size mismatch errors. Allows selecting cropping or upsampling methods as needed.  

---

### CogVideoX FasterCache

| Parameter Name        | Type    | Default Value   | Parameter Nature | Function Description                                                                                     |
| :-------------------- | :------ | :------------ | :------- | :------------------------------------------------------------------------------------------------------- |
| `start_step`          | INT     | 15              | Optional | The step from which to start enabling cache reuse, skipping initial computations to save VRAM and accelerate subsequent inference. |
| `hf_step`             | INT     | —               | Optional | High-frequency feature cache interval: how often to reuse high-frequency features.                         |
| `lf_step`             | INT     | —               | Optional | Low-frequency feature cache interval: how often to reuse low-frequency features.                           |
| `cache_device`        | STRING  | `"main_device"` | Optional | Device for cache storage, selectable from `"main_device"`, `"offload_device"`, or e.g., `"cuda:1"`.        |
| `num_blocks_to_cache` | INT     | —               | Optional | Number of Transformer Blocks to cache, controlling cache granularity.                                    |

**Outputs**  
- `FASTERCACHEARGS`: A configuration object encapsulating all cache parameters, ready for direct input to samplers or model inference functions.  

**Use Cases**  
- For long-sequence video generation tasks, significantly reduces redundant computations by skipping initial steps and reusing high/low-frequency features in later stages, thereby decreasing VRAM usage and boosting overall inference speed.  

---

### CogVideo TorchCompileSettings

| Parameter Name           | Type    | Default Value | Parameter Nature | Function Description                                                                                                     |
| :----------------------- | :------ | :------------ | :------- | :----------------------------------------------------------------------------------------------------------------------- |
| `backend`                | STRING  | `"inductor"`  | Optional | Specifies the backend used by `torch.compile`, selectable from `"inductor"` (default), `"nvfuser"`, and other custom backends. |
| `mode`                   | STRING  | `"default"`   | Optional | Compilation mode, selectable from `"default"`, `"reduce-overhead"`, `"max-autotune"`, etc., controlling optimization intensity and strategy. |
| `fullgraph`              | BOOLEAN | `False`       | Optional | Whether to enable full-graph mode; if `True`, the entire model is captured as a single graph, otherwise it might error on graph breaks. |
| `dynamic`                | BOOLEAN | `False`       | Optional | Whether to enable dynamic shape support, generating more general kernels to reduce recompilation due to input size changes. |
| `dynamo_cache_size_limit`| INT     | `8`           | Optional | Sets `torch._dynamo.config.cache_size_limit` (default 8), controlling the maximum number of compiled cache versions a single function can generate, preventing infinite recompilation. |

**Outputs**  
- `torch_compile_args`: Encapsulates the actual effective compilation parameters, including `backend`, `mode`, `fullgraph`, `dynamic`, and `dynamo_cache_size_limit`.  

**Use Cases**  
- Significantly accelerates model execution using PyTorch 2.0+'s `torch.compile` in scenarios with ample GPU resources and demanding inference speed requirements.  
- Flexibly switches backends and modes based on different hardware characteristics and model structures to achieve the best performance-stability balance.  
- During debugging, can adjust `dynamic` and `fullgraph` parameters to identify and resolve compilation failures or performance bottlenecks.  

---

### CogVideo Context Options

| Parameter Name     | Type   | Default Value          | Parameter Nature | Function Description                                                                 |
| :----------------- | :----- | :--------------------- | :------- | :----------------------------------------------------------------------------------- |
| `context_schedule` | STRING | `"uniform_standard"`   | Optional | Context scheduling strategy, selectable: <br>• `uniform_standard` <br>• `uniform_looped` <br>• `static_standard`. |
| `context_frames`   | INT    | `32`                   | Optional | Maximum number of context frames to retain; earliest frames are dropped if input frames exceed this value. |
| `context_stride`   | INT    | `1`                    | Optional | Stride for extracting context frames from the original sequence, controlling frame intervals. |
| `context_overlap`  | INT    | `0`                    | Optional | Number of overlapping frames between adjacent context windows, for smooth transitions. |
| `freenoise`        | BOOLEAN | `False`                | Optional | Whether to enable FreeNoise noise shuffling mechanism, periodically introducing random perturbations in context frames. |

**Outputs**  
- `COGCONTEXT`: A dictionary containing all effective context configuration parameters.  

**Use Cases**  
- In long-sequence video generation like Vid2Vid or Pose2Vid, balances VRAM usage and generation coherence by limiting retained frames, setting stride and overlap, and optionally applying noise perturbation.  
- Flexibly adjusts scheduling strategies (e.g., looped vs. static) and noise mechanisms for different scenarios to optimize motion coherence or increase visual diversity.  

---

### CogVideo LatentPreview

| Parameter Name | Type   | Default Value | Parameter Nature | Function Description                                                    |
| :------------- | :----- | :------------ | :------- | :---------------------------------------------------------------------- |
| `samples`      | LATENT | —             | Required | Input video latent tensor, with shape `[batch, num_frames, channels, H, W]`. |
| `seed`         | INT    | —             | Optional | Random seed used for preview generation, ensuring reproducibility.        |
| `min_val`      | FLOAT  | —             | Optional | Minimum latent value mapped during visualization; values below this will be clipped. |
| `max_val`      | FLOAT  | —             | Optional | Maximum latent value mapped during visualization; values above this will be clipped. |
| `r_bias`       | FLOAT  | —             | Optional | Red channel offset, for adjusting the red component in the preview image. |
| `g_bias`       | FLOAT  | —             | Optional | Green channel offset, for adjusting the green component in the preview image. |
| `b_bias`       | FLOAT  | —             | Optional | Blue channel offset, for adjusting the blue component in the preview image. |

**Outputs**  
- `IMAGE`: Preview image, displaying the visualization effect of the specified latent frame.  
- `STRING`: Text information, including current `seed`, `min_val`/`max_val` range, and channel offsets.  

**Use Cases**  
- Real-time visualization of intermediate latent representations in the video generation workflow, aiding in debugging, calibrating mapping ranges, and color offset parameters to quickly identify and optimize generation effects.  

---

### CogVideo Enhance-A-Video

| Parameter Name    | Type    | Default Value | Parameter Nature | Function Description                                                          |
| :---------------- | :------ | :------------ | :------- | :---------------------------------------------------------------------------- |
| `weight`          | FLOAT   | 1.0           | Optional | Enhancement temperature factor, multiplies cross-frame attention strength to improve video coherence and detail expression.    |
| `start_percent`   | FLOAT   | 0.0           | Optional | Percentage from the video start to apply enhancement (0.0–1.0).               |
| `end_percent`     | FLOAT   | 1.0           | Optional | Percentage to the video end to stop enhancement (0.0–1.0).                    |

**Outputs**  
- `FETAARGS`: Data structure containing enhanced cross-frame attention adjustment parameters, usable in subsequent decoding or rendering.  

**Use Cases**  
- After the generation process, performs training-free fine-tuning on the video's temporal attention output to improve video details and visual coherence, especially suitable for videos with frequent character movements or scene changes.  

### CogVideo LoraSelect

| Parameter Name | Type    | Default Value | Parameter Nature | Function Description                                         |
| :------------- | :------ | :------------ | :------- | :----------------------------------------------------------- |
| `model`        | MODEL   | —             | Required | Input CogVideo model or pipeline object.                     |
| `lora_path`    | STRING  | —             | Required | Local file system path or remote URL for the LoRA weight file. |
| `lora_scale`   | FLOAT   | 1.0           | Optional | LoRA weight application strength (0.0–1.0), controlling effect intensity. |
| `unet_only`    | BOOLEAN | False         | Optional | Whether to apply LoRA only to the UNet submodule.            |

**Outputs**  
- `model`: Model object with the specified LoRA weights loaded and applied.  

**Use Cases**  
- Dynamically introduces custom LoRA weights during video generation to achieve style fine-tuning or special effect enhancement.  

---

### CogVideo LoraSelect Comfy

| Parameter Name | Type    | Default Value | Parameter Nature | Function Description                                                    |
| :------------- | :------ | :------------ | :------- | :---------------------------------------------------------------------- |
| `model`        | MODEL   | —             | Required | Input CogVideo model or pipeline object.                                |
| `lora_name`    | STRING  | —             | Required | Name of the preset LoRA weight stored in ComfyUI's default LoRA directory. |
| `strength`     | FLOAT   | 1.0           | Optional | LoRA weight application strength (0.0–1.0), controlling render effect proportion. |
| `overwrite`    | BOOLEAN | False         | Optional | Whether to overwrite all existing LoRA deltas in the model (`True` to overwrite, `False` to stack). |

**Outputs**  
- `model`: Model object with ComfyUI's preset LoRA weights loaded and applied.  

**Use Cases**  
- Utilizes ComfyUI's native LoRA management system for quick switching and application of preset weights without manual path specification.  

---

### CogVideo TransformerEdit

| Parameter Name  | Type   | Default Value | Parameter Nature | Function Description                                              |
| :-------------- | :------ | :------------ | :------- | :---------------------------------------------------------------- |
| `remove_blocks` | STRING | “”            | Required | Comma-separated list of Transformer Block indices, e.g., `"15,25,37"`. |

**Outputs**  
- `block_list`: Integer list, containing the indices of the actually removed Blocks.  

**Use Cases**  
- Precisely prunes model layers to reduce VRAM usage or accelerate inference; used for experimental layer comparison.  

---

### Tora Encode Trajectory

| Parameter Name | Type      | Default Value | Parameter Nature | Function Description                                           |
| :------------- | :-------- | :------------ | :------- | :------------------------------------------------------------- |
| `trajectory`   | PATH      | —             | Required | User-drawn or imported motion trajectory file (SVG, JSON, etc.). |
| `resolution`   | STRING    | `"512x512"`   | Optional | Spatial resolution (width×height) of the generated latent patch. |
| `frame_count`  | INTEGER   | 16            | Optional | Number of motion patch frames to generate.                     |

**Outputs**  
- `trajectory_embeds`: Spatiotemporal motion patch latent vectors, ready for direct input to sampling nodes.  

**Use Cases**  
- In I2V generation workflows, controls motion according to custom trajectories; suitable for animation production and path-driven effects.  

## 🔧 Common Workflow Combinations

| Workflow Name                      | Node Combination                                          | Purpose                                                                 |
| :--------------------------------- | :-------------------------------------------------------- | :---------------------------------------------------------------------- |
| Text-to-Video Generation           | CogVideo TextEncode → CogVideo Sampler → CogVideo Decode | Generates video content based on text prompts (core function, highest stability). |
| Image-to-Video Conversion          | CogVideo ImageEncode → CogVideo Sampler → CogVideo Decode | Expands static images into dynamic videos (requires image encoder compatibility verification). |
| Trajectory-Controlled Video Generation | Tora Encode Trajectory → CogVideo Sampler → CogVideo Decode | Controls object motion paths via trajectory coordinates (relies on trajectory encoder precision). |
| Accelerated Video Generation       | CogVideo FasterCache → CogVideo Sampler → CogVideo Decode | Improves generation speed through VRAM optimization (actual speedup 20%-30%). |
| Context Window Adjustment          | CogVideo Context Options → CogVideo Sampler → CogVideo Decode | Adjusts temporal context length (effective in 16-64 frame range).         |
| Video Enhancement Processing       | CogVideo Decode → CogVideo Enhance-A-Video              | Resolution upscaling/frame interpolation enhancement (requires separate enhancement module deployment). |