# Awesome-ComfyUI-Video

> For the Chinese version of this document, please see [README.zh.md](README.zh.md).

## Table of Contents

- [Awesome-ComfyUI-Video](#awesome-comfyui-video)
  - [Table of Contents](#table-of-contents)
  - [Official Video Nodes](#official-video-nodes)
  - [Outstanding Community Custom Nodes](#outstanding-community-custom-nodes)
  - [Example Video Workflows](#example-video-workflows)

## Official Video Nodes

| Name                | Description                | Main Usage/Highlight      | Docs/Link |
| ------------------- | -------------------------- | ------------------------- | --------- |
| Load Video          | Load video files as input  | Video input for workflows | [Docs](#) |
| Save Video          | Save output as video files | Export results as video   | [Docs](#) |
| Video Frame Extract | Extract frames from video  | Frame-level processing    | [Docs](#) |
| ...                 | ...                        | ...                       | ...       |

## Outstanding Community Custom Nodes

| Name                        | Author       | Description                                                                                                                                                                                                                                                                             | Stars | Last Update         | Repo                                                              | Details Doc                                            |
| --------------------------- | ------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----- | ------------------- | ----------------------------------------------------------------- | ------------------------------------------------------ |
| ComfyUI-3D-Pack             | MrForExample | Make 3D assets generation in ComfyUI good and convenient as it generates image/video! This is an extensive node suite that enables ComfyUI to process 3D inputs (Mesh & UV Texture, etc.) using cutting edge algorithms (3DGS, NeRF, etc.) and models (InstantMesh, CRM, TripoSR, etc.) | 3007  | 2025-01-24 18:41:37 | [GitHub](https://github.com/MrForExample/ComfyUI-3D-Pack)         | ...                                                    |
| ComfyUI-HunyuanVideoWrapper | kijai        | ComfyUI diffusers wrapper nodes for [HunyuanVideo](https://github.com/Tencent/HunyuanVideo)                                                                                                                                                                                             | 2350  | 2025-03-30 16:48:09 | [GitHub](https://github.com/kijai/ComfyUI-HunyuanVideoWrapper)    | [Details](docs/en/custom_nodes/hunyuanvideowrapper.md)                                                    |
| EasyAnimate                 | bubbliiiing  | Video Generation Nodes for EasyAnimate, which suppors text-to-video, image-to-video, video-to-video and different controls.                                                                                                                                                             | 2130  | 2025-03-06 11:41:28 | [GitHub](https://github.com/aigc-apps/EasyAnimate)                | [Details](docs/en/custom_nodes/easyanimate.md)                                                    |
| comfyui-mixlab-nodes        | shadowcz007  | 3D, ScreenShareNode & FloatingVideoNode, SpeechRecognition & SpeechSynthesis, GPT, LoadImagesFromLocal, Layers, Other Nodes, ...                                                                                                                                                        | 1574  | 2025-02-05 10:24:45 | [GitHub](https://github.com/shadowcz007/comfyui-mixlab-nodes)     | ...                                                    |
| ComfyUI-CogVideoXWrapper    | kijai        | Diffusers wrapper for CogVideoX -models: [CogVideo](https://github.com/THUDM/CogVideo)                                                                                                                                                                                                  | 1476  | 2025-02-17 00:48:16 | [GitHub](https://github.com/kijai/ComfyUI-CogVideoXWrapper)       | ...                                                    |
| ComfyUI-LTXVideo            | lightricks   | Custom nodes for LTX-Video support in ComfyUI                                                                                                                                                                                                                                           | 1036  | 2025-04-17 15:21:00 | [GitHub](https://github.com/Lightricks/ComfyUI-LTXVideo)          | ...                                                    |
| ComfyUI-VideoHelperSuite    | Kosinkadink  | Nodes related to video workflows                                                                                                                                                                                                                                                        | 939   | 2025-04-18 18:54:23 | [GitHub](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) | [Details](docs/en/custom_nodes/videohelpersuite.md)    |
| VideoX-Fun                  | bubbliiiing  | VideoX-Fun is a video generation pipeline that can be used to generate AI images and videos, as well as to train baseline and Lora models for Diffusion Transformer.                                                                                                                    | 914   | 2025-04-18 03:01:53 | [GitHub](https://github.com/aigc-apps/VideoX-Fun)                 | ...                                                    |
| Steerable Motion            | banodoco     | Steerable Motion is a ComfyUI node for batch creative interpolation. Our goal is to feature the best methods for steering motion with images as video models evolve.                                                                                                                    | 881   | 2024-06-15 23:01:54 | [GitHub](https://github.com/banodoco/steerable-motion)            | ...                                                    |
| ComfyUI-segment-anything-2  | kijai        | Nodes to use [segment-anything-2](https://github.com/facebookresearch/segment-anything-2) for image or video segmentation.                                                                                                                                                              | 867   | 2025-03-19 09:40:37 | [GitHub](https://github.com/kijai/ComfyUI-segment-anything-2)     | ...                                                    |

- 👉 [See all node packages and details in nodes.md (future English version might be at docs/en/nodes_en.md)](docs/nodes.md)

## Example Video Workflows
Showcase several typical node combination workflows, focusing on real application scenarios.

- **Workflow Name**  
  - Use case
  - Involved nodes
  - Brief process description
  - [Optional: diagram or link]

---

> Each section is recommended to only list the most commonly used or representative content. For more details, use "see more" or links to avoid excessive length in the main text.