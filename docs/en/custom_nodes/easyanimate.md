# 🎥 EasyAnimate

## 🔍 Overview

| Item         | Details                                                                 |
| ------------ | ---------------------------------------------------------------------- |
| 📌 **Author** | [@aigc-apps](https://github.com/aigc-apps/EasyAnimate)                 |
| 📅 **Version** | 5.1+                                                                  |
| 🏷️ **Category** | Video Generation and Control                                         |
| 🔗 **Repository** | [GitHub Link](https://github.com/aigc-apps/EasyAnimate)             |

## 📝 Introduction
EasyAnimate is a powerful video generation tool that supports generating high-quality videos from text, images, or videos, with various control features (e.g., camera trajectory, depth, pose).

## 📊 Node List

| Node Name                        | Type          | Main Function                              | Complexity | Common Use Cases          |
| -------------------------------- | ------------- | ------------------------------------------ | ---------- | ------------------------- |
| **Load EasyAnimate Model**       | Model Loading | Load EasyAnimate model                     | ★★☆☆☆      | Model initialization      |
| **Load EasyAnimate Lora**        | Model Loading | Load Lora weights for EasyAnimate          | ★★☆☆☆      | Model fine-tuning         |
| **EasyAnimate_TextBox**          | Input         | Provides a text input box                  | ★☆☆☆☆      | Inputting prompts         |
| **EasyAnimate Sampler for T2V**  | Sampler       | Text-to-Video sampling                     | ★★★☆☆      | Text-to-video generation  |
| **EasyAnimateV5 Sampler for T2V**| Sampler       | Text-to-Video sampling (V5+)               | ★★★☆☆      | Text-to-video (V5+)       |
| **EasyAnimate Sampler for I2V**  | Sampler       | Image-to-Video sampling                    | ★★★☆☆      | Image-to-video generation |
| **EasyAnimateV5 Sampler for I2V**| Sampler       | Image-to-Video sampling (V5+)              | ★★★☆☆      | Image-to-video (V5+)      |
| **EasyAnimate Sampler for V2V**  | Sampler       | Video-to-Video sampling                    | ★★★☆☆      | Video style transfer      |
| **EasyAnimateV5 Sampler for V2V**| Sampler       | Video-to-Video sampling (V5+)              | ★★★☆☆      | Video style transfer (V5+) |
| **Create Trajectory Based On KJNodes** | Utility | Create trajectory based on KJNodes         | ★★☆☆☆      | Camera trajectory control |
| **CameraBasicFromChaoJie**       | Utility       | Basic camera control                       | ★★☆☆☆      | Simple camera motion      |
| **CameraTrajectoryFromChaoJie** | Utility       | Generate camera trajectory parameters      | ★★☆☆☆      | Advanced camera motion    |
| **CameraCombineFromChaoJie**    | Utility       | Combine multiple camera motions            | ★★☆☆☆      | Complex camera motion     |
| **CameraJoinFromChaoJie**       | Utility       | Join multiple camera trajectories          | ★★☆☆☆      | Trajectory stitching      |
| **ImageMaximumNode**            | Utility       | Merge two video frames                     | ★★☆☆☆      | Frame merging             |

## 📑 Core Node Details

### Load EasyAnimate Model

**Description:**
Load the EasyAnimate model, supporting various model types and precisions.

**Input Parameters:**

| Parameter Name     | Type   | Default Value            | Property  | Description                     |
| ------------------ | ------ | ------------------------ | --------- | ------------------------------- |
| model              | STRING | EasyAnimateV5.1-12b-zh  | Required  | Name of the model to load       |
| GPU_memory_mode    | STRING | model_cpu_offload       | Optional  | GPU memory mode                 |
| model_type         | STRING | Inpaint                | Optional  | Model type (e.g., Inpaint, Control) |
| config             | STRING | easyanimate_video_v5.1_magvit_qwen.yaml | Required | Path to the model config file   |
| precision          | STRING | bf16                   | Optional  | Model precision (e.g., fp16, bf16) |

**Outputs:**

* `EASYANIMATESMODEL`: The loaded EasyAnimate model object.

**Use Cases:**
- Initialize the EasyAnimate model for video generation.

---

### Load EasyAnimate Lora

**Description:**
Load Lora weights for EasyAnimate, used for model fine-tuning.

**Input Parameters:**

| Parameter Name     | Type   | Default Value | Property  | Description                     |
| ------------------ | ------ | ------------- | --------- | ------------------------------- |
| easyanimate_model  | EASYANIMATESMODEL | -         | Required  | Loaded EasyAnimate model object |
| lora_name          | STRING | -             | Required  | Lora weight file name           |
| strength_model     | FLOAT  | 1.0           | Optional  | Lora weight strength            |
| lora_cache         | BOOLEAN| False         | Optional  | Enable Lora cache               |

**Outputs:**

* `EASYANIMATESMODEL`: The model object with Lora weights loaded.

**Use Cases:**
- Fine-tune the model for specific tasks.

---

### EasyAnimate_TextBox

**Description:**
Provides a multi-line text input box, typically used for inputting prompts.

**Input Parameters:**

| Parameter Name | Type   | Default Value | Property | Description        |
| -------------- | ------ | ------------- | -------- | ------------------ |
| prompt         | STRING | ""            | Required | Text input content |

**Outputs:**

* `prompt` (STRING_PROMPT): The output text content, connectable to the prompt input of sampler nodes.

**Use Cases:**
- Providing positive or negative prompts for EasyAnimate samplers.

---

### EasyAnimate Sampler for T2V

**Description:**
Generate videos from text.

**Input Parameters:**

| Parameter Name     | Type   | Default Value | Property  | Description                     |
| ------------------ | ------ | ------------- | --------- | ------------------------------- |
| easyanimate_model  | EASYANIMATESMODEL | -         | Required  | Loaded EasyAnimate model object |
| prompt             | STRING | -             | Required  | Text prompt                     |
| negative_prompt    | STRING | -             | Optional  | Negative prompt                 |
| video_length       | INT    | 72            | Required  | Number of video frames          |
| width              | INT    | 1008          | Required  | Video width                     |
| height             | INT    | 576           | Required  | Video height                    |
| is_image           | BOOLEAN| False         | Optional  | Whether to generate a single image |
| seed               | INT    | 43            | Optional  | Random seed                     |
| steps              | INT    | 25            | Required  | Sampling steps                  |
| cfg                | FLOAT  | 7.0           | Required  | Guidance strength               |
| scheduler          | STRING | DDIM          | Required  | Sampler type (Options: Euler, Euler A, DPM++, PNDM, DDIM) |

**Outputs:**

* `IMAGE`: Generated video frames.

**Use Cases:**
- Generate video content based on text descriptions.

---

### EasyAnimateV5 Sampler for T2V

**Description:**
Generate videos from text (EasyAnimate V5+ version optimized). This version typically includes specific optimizations and parameters for V5 series models, such as `teacache`.

**Input Parameters:**

| Parameter Name       | Type   | Default Value | Property | Description                     |
| -------------------- | ------ | ------------- | -------- | ------------------------------- |
| easyanimate_model    | EASYANIMATESMODEL | -      | Required | Loaded EasyAnimate model object |
| prompt               | STRING_PROMPT | -      | Required | Text prompt                     |
| negative_prompt      | STRING_PROMPT | -      | Optional | Negative prompt                 |
| video_length         | INT    | 49     | Required | Number of video frames (V5 usually 1-49) |
| width                | INT    | 1008   | Required | Video width                     |
| height               | INT    | 576    | Required | Video height                    |
| is_image             | BOOLEAN| False  | Optional | Whether to generate a single image |
| seed                 | INT    | 43     | Optional | Random seed                     |
| steps                | INT    | 25     | Required | Sampling steps                  |
| cfg                  | FLOAT  | 7.0    | Required | Guidance strength               |
| scheduler            | COMBO  | Flow   | Required | Sampler type (Flow recommended for V5) |
| teacache_threshold   | FLOAT  | 0.10   | Optional | TeaCache threshold              |
| enable_teacache      | BOOLEAN| True   | Optional | Enable TeaCache                 |

**Outputs:**

* `IMAGE`: Generated video frames.

**Use Cases:**
- Text-to-video generation using EasyAnimate V5 and newer models, leveraging their specific optimizations.

---

### EasyAnimate Sampler for I2V

**Description:**
Generate videos from images.

**Input Parameters:**

| Parameter Name     | Type   | Default Value | Property  | Description                     |
| ------------------ | ------ | ------------- | --------- | ------------------------------- |
| easyanimate_model  | EASYANIMATESMODEL | -         | Required  | Loaded EasyAnimate model object |
| prompt             | STRING | -             | Required  | Text prompt                     |
| negative_prompt    | STRING | -             | Optional  | Negative prompt                 |
| video_length       | INT    | 72            | Required  | Number of video frames          |
| base_resolution    | COMBO  | 768           | Required  | Base resolution (512, 768, 960, 1024) |
| seed               | INT    | 43            | Optional  | Random seed                     |
| steps              | INT    | 25            | Required  | Sampling steps                  |
| cfg                | FLOAT  | 7.0           | Required  | Guidance strength               |
| scheduler          | STRING | DDIM          | Required  | Sampler type (Options: Euler, Euler A, DPM++, PNDM, DDIM) |
| start_img          | IMAGE  | -             | Optional  | Starting image                  |
| end_img            | IMAGE  | -             | Optional  | Ending image                    |

**Outputs:**

* `IMAGE`: Generated video frames.

**Use Cases:**
- Generate continuous video content from images.

---

### EasyAnimateV5 Sampler for I2V

**Description:**
Generate videos from images (EasyAnimate V5+ version optimized). This version typically includes specific optimizations and parameters for V5 series models, such as `teacache`.

**Input Parameters:**

| Parameter Name       | Type   | Default Value | Property | Description                     |
| -------------------- | ------ | ------------- | -------- | ------------------------------- |
| easyanimate_model    | EASYANIMATESMODEL | -      | Required | Loaded EasyAnimate model object |
| prompt               | STRING_PROMPT | -      | Required | Text prompt                     |
| negative_prompt      | STRING_PROMPT | -      | Optional | Negative prompt                 |
| video_length         | INT    | 49     | Required | Number of video frames (V5 usually 1-49) |
| base_resolution      | COMBO  | 768    | Required | Base resolution (512, 768, 960, 1024) |
| seed                 | INT    | 43     | Optional | Random seed                     |
| steps                | INT    | 25     | Required | Sampling steps                  |
| cfg                  | FLOAT  | 7.0    | Required | Guidance strength               |
| scheduler            | COMBO  | Flow   | Required | Sampler type (Flow recommended for V5) |
| teacache_threshold   | FLOAT  | 0.10   | Optional | TeaCache threshold              |
| enable_teacache      | BOOLEAN| True   | Optional | Enable TeaCache                 |
| start_img            | IMAGE  | -      | Optional | Starting image                  |
| end_img              | IMAGE  | -      | Optional | Ending image                    |

**Outputs:**

* `IMAGE`: Generated video frames.

**Use Cases:**
- Image-to-video generation using EasyAnimate V5 and newer models, leveraging their specific optimizations.

---

### EasyAnimate Sampler for V2V

**Description:**
Generate videos from videos.

**Input Parameters:**

| Parameter Name     | Type   | Default Value | Property  | Description                     |
| ------------------ | ------ | ------------- | --------- | ------------------------------- |
| easyanimate_model  | EASYANIMATESMODEL | -         | Required  | Loaded EasyAnimate model object |
| prompt             | STRING | -             | Required  | Text prompt                     |
| negative_prompt    | STRING | -             | Optional  | Negative prompt                 |
| video_length       | INT    | 72            | Required  | Number of video frames          |
| base_resolution    | COMBO  | 768           | Required  | Base resolution (512, 768, 960, 1024) |
| seed               | INT    | 43            | Optional  | Random seed                     |
| steps              | INT    | 25            | Required  | Sampling steps                  |
| cfg                | FLOAT  | 7.0           | Required  | Guidance strength               |
| denoise_strength   | FLOAT  | 0.70          | Required  | Denoising strength              |
| scheduler          | STRING | DDIM          | Required  | Sampler type (Options: Euler, Euler A, DPM++, PNDM, DDIM) |
| validation_video   | IMAGE  | -             | Optional  | Input validation video          |
| control_video      | IMAGE  | -             | Optional  | Input control video             |

**Outputs:**

* `IMAGE`: Generated video frames.

**Use Cases:**
- Perform style transfer or enhancement on input videos.

---

### EasyAnimateV5 Sampler for V2V

**Description:**
Generate videos from videos (optimized for EasyAnimate V5+ versions). This version typically includes specific optimizations and parameters for V5 series models, such as `teacache`, and supports more complex controls like camera conditions.

**Input Parameters:**

| Parameter Name       | Type   | Default Value | Property | Description                     |
| -------------------- | ------ | ------------- | -------- | ------------------------------- |
| easyanimate_model    | EASYANIMATESMODEL | -      | Required | Loaded EasyAnimate model object |
| prompt               | STRING_PROMPT | -      | Required | Text prompt                     |
| negative_prompt      | STRING_PROMPT | -      | Optional | Negative prompt                 |
| video_length         | INT    | 49     | Required | Number of video frames (V5 usually 1-49) |
| base_resolution      | COMBO  | 768    | Required | Base resolution (512, 768, 960, 1024) |
| seed                 | INT    | 43     | Optional | Random seed                     |
| steps                | INT    | 25     | Required | Sampling steps                  |
| cfg                  | FLOAT  | 7.0    | Required | Guidance strength               |
| denoise_strength     | FLOAT  | 0.70   | Required | Denoising strength              |
| scheduler            | COMBO  | Flow   | Required | Sampler type (Flow recommended for V5) |
| teacache_threshold   | FLOAT  | 0.10   | Optional | TeaCache threshold              |
| enable_teacache      | BOOLEAN| True   | Optional | Enable TeaCache                 |
| validation_video     | IMAGE  | -      | Optional | Input validation video (for Inpaint models) |
| control_video        | IMAGE  | -      | Optional | Input control video (for Control models) |
| camera_conditions    | STRING | -      | Optional | Camera motion conditions (JSON string) |
| ref_image            | IMAGE  | -      | Optional | Reference image                 |

**Outputs:**

* `IMAGE`: Generated video frames.

**Use Cases:**
- Video-to-video style transfer or content modification using EasyAnimate V5 and newer models, with support for camera control.

---

### Create Trajectory Based On KJNodes

**Description:**
Creates trajectory images based on KJNodes' coordinate and mask inputs, for trajectory control in EasyAnimate. Trajectory points are drawn as Gaussian heatmaps on the image.

**Input Parameters:**

| Parameter Name | Type   | Default Value | Property | Description                                      |
| -------------- | ------ | ------------- | -------- | ------------------------------------------------ |
| coordinates    | STRING | -             | Required | Trajectory coordinates (List of JSON strings or single JSON string) |
| masks          | MASK   | -             | Required | Trajectory mask (determines output image size)   |

**Outputs:**

* `IMAGE`: Generated trajectory image sequence.

**Use Cases:**
- Providing trajectory control maps for EasyAnimate's Control models.
- Defining complex motion paths in conjunction with KJNodes' curve tools.

---

### CameraBasicFromChaoJie

**Description:**
Generates basic camera motion.

**Input Parameters:**

| Parameter Name | Type   | Default Value | Property | Description                     |
| -------------- | ------ | ------------- | -------- | ------------------------------- |
| camera_pose    | STRING | Static        | Required | Camera motion type              |
| speed          | FLOAT  | 1.0           | Required | Camera motion speed             |
| video_length   | INT    | 16            | Required | Number of video frames          |

**Outputs:**

* `CameraPose`: Camera motion trajectory.

**Use Cases:**
- Create simple camera motion effects.

---

### CameraTrajectoryFromChaoJie

**Description:**
Converts a `CameraPose` object into a JSON string of camera trajectory parameters and video length, for camera control in EasyAnimate.

**Input Parameters:**

| Parameter Name | Type       | Default Value | Property | Description          |
| -------------- | ---------- | ------------- | -------- | -------------------- |
| camera_pose    | CameraPose | -             | Required | Camera motion pose object |
| fx             | FLOAT      | 0.474812461   | Optional | Focal length x-axis  |
| fy             | FLOAT      | 0.844111024   | Optional | Focal length y-axis  |
| cx             | FLOAT      | 0.5           | Optional | Optical center x-axis |
| cy             | FLOAT      | 0.5           | Optional | Optical center y-axis |

**Outputs:**

* `camera_trajectory` (STRING): JSON string of camera trajectory parameters.
* `video_length` (INT): Video length (number of frames).

**Use Cases:**
- Converting camera poses generated by `CameraBasicFromChaoJie`, `CameraCombineFromChaoJie`, or `CameraJoinFromChaoJie` into `camera_conditions` input usable by EasyAnimate V5+ V2V sampler.

---

### CameraCombineFromChaoJie

**Description:**
Combines multiple camera motion trajectories.

**Input Parameters:**

| Parameter Name | Type   | Default Value | Property | Description                     |
| -------------- | ------ | ------------- | -------- | ------------------------------- |
| camera_pose1   | STRING | Static        | Required | First camera motion trajectory  |
| camera_pose2   | STRING | Static        | Required | Second camera motion trajectory |
| camera_pose3   | STRING | Static        | Optional | Third camera motion trajectory  |
| camera_pose4   | STRING | Static        | Optional | Fourth camera motion trajectory |
| speed          | FLOAT  | 1.0           | Required | Camera motion speed             |
| video_length   | INT    | 16            | Required | Number of video frames          |

**Outputs:**

* `CameraPose`: Combined camera motion trajectory.

**Use Cases:**
- Create complex camera motion effects.

---

### CameraJoinFromChaoJie

**Description:**
Joins two camera motion pose (`CameraPose`) objects to create a new, combined camera motion pose.

**Input Parameters:**

| Parameter Name | Type       | Default Value | Property | Description              |
| -------------- | ---------- | ------------- | -------- | ------------------------ |
| camera_pose1   | CameraPose | -             | Required | First camera motion pose |
| camera_pose2   | CameraPose | -             | Required | Second camera motion pose|

**Outputs:**

* `CameraPose`: The joined camera motion pose object.

**Use Cases:**
- Sequentially connecting multiple simple camera movements to form more complex camera motion sequences.
- For example, a pan followed by a rotation.

---

### ImageMaximumNode

**Description:**
Merges two video frame sequences, taking the maximum value for each pixel at each frame.

**Input Parameters:**

| Parameter Name | Type   | Default Value | Property | Description              |
| -------------- | ------ | ------------- | -------- | ------------------------ |
| video_1        | IMAGE  | -             | Required | First video frame sequence |
| video_2        | IMAGE  | -             | Required | Second video frame sequence|

**Outputs:**

* `IMAGE`: Merged video frame sequence.

**Use Cases:**
- Merge two video frame sequences to create special effects.

---

## 🔧 Common Workflow Combinations

| Workflow Name             | Node Combination                                                                    | Purpose                                  |
| ------------------------- | ----------------------------------------------------------------------------------- | ---------------------------------------- |
| Text-to-Video Generation  | Load EasyAnimate Model → EasyAnimate_TextBox → EasyAnimate Sampler for T2V            | Generate videos from text                |
| V5+ Text-to-Video         | Load EasyAnimate Model → EasyAnimate_TextBox → EasyAnimateV5 Sampler for T2V          | V5+ model text-to-video generation       |
| Image-to-Video Generation | Load EasyAnimate Model → EasyAnimate_TextBox → EasyAnimate Sampler for I2V            | Generate videos from images              |
| V5+ Image-to-Video        | Load EasyAnimate Model → EasyAnimate_TextBox → EasyAnimateV5 Sampler for I2V          | V5+ model image-to-video generation      |
| Video Style Transfer      | Load EasyAnimate Model → EasyAnimate_TextBox → EasyAnimate Sampler for V2V            | Perform style transfer on videos         |
| V5+ Video Style Transfer (Camera Ctrl) | Load EasyAnimate Model → EasyAnimate_TextBox → CameraBasic/Combine/Join → CameraTrajectory → EasyAnimateV5 Sampler for V2V | V5+ model video transfer with camera control |
| V5+ Video Style Transfer (Trajectory Ctrl) | Load EasyAnimate Model → EasyAnimate_TextBox → [KJNodes for Coords/Masks] → Create Trajectory → EasyAnimateV5 Sampler for V2V | V5+ model video transfer with trajectory control |
| Complex Camera Motion     | CameraBasicFromChaoJie → CameraJoinFromChaoJie → CameraCombineFromChaoJie → CameraTrajectoryFromChaoJie | Create complex camera motion parameters |
