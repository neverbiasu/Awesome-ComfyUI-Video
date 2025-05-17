# Awesome-ComfyUI-Video (精选ComfyUI视频节点集合)

> 本文档的英文版本，请参阅 [README.md](README.md)。

## 目录

- [官方视频节点](#官方视频节点)
- [优秀社区自定义节点](#优秀社区自定义节点)
- [示例视频工作流](#示例视频工作流)

## 官方视频节点

| 名称                | 描述                       | 主要用途/亮点             | 文档/链接 |
| ------------------- | -------------------------- | ------------------------- | --------- |
| Load Video          | 加载视频文件作为输入         | 工作流的视频输入          | [文档](#)  |
| Save Video          | 将输出保存为视频文件         | 将结果导出为视频          | [文档](#)  |
| Video Frame Extract | 从视频中提取帧             | 帧级别处理                | [文档](#)  |
| ...                 | ...                        | ...                       | ...       |

## 优秀社区自定义节点

| 名称                        | 作者         | 描述                                                                                                                                                                                                                                                                                | Stars | 最后更新时间        | 仓库                                                                  | 详细文档                                                     |
| --------------------------- | ------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----- | ------------------- | ----------------------------------------------------------------- | ------------------------------------------------------------ |
| ComfyUI-3D-Pack             | MrForExample | 使ComfyUI中的3D资产生成像生成图像/视频一样出色和便捷！这是一个广泛的节点套件，使ComfyUI能够使用尖端算法（3DGS、NeRF等）和模型（InstantMesh、CRM、TripoSR等）处理3D输入（网格和UV纹理等）。                                                                                             | 3007  | 2025-01-24 18:41:37 | [GitHub](https://github.com/MrForExample/ComfyUI-3D-Pack)         | ...                                                          |
| ComfyUI-HunyuanVideoWrapper | kijai        | [混元DiT](https://github.com/Tencent/HunyuanVideo) 的ComfyUI Diffusers封装节点。                                                                                                                                                                                                        | 2350  | 2025-03-30 16:48:09 | [GitHub](https://github.com/kijai/ComfyUI-HunyuanVideoWrapper)    | ...                                                          |
| EasyAnimate                 | bubbliiiing  | EasyAnimate的视频生成节点，支持文本到视频、图像到视频、视频到视频以及不同的控制。                                                                                                                                                                                                             | 2130  | 2025-03-06 11:41:28 | [GitHub](https://github.com/aigc-apps/EasyAnimate)                | [Details](docs/zh/custom_nodes/easyanimate.md)                                                          |
| comfyui-mixlab-nodes        | shadowcz007  | 3D、ScreenShareNode & FloatingVideoNode、语音识别与合成、GPT、从本地加载图像、图层、其他节点等...                                                                                                                                                                                             | 1574  | 2025-02-05 10:24:45 | [GitHub](https://github.com/shadowcz007/comfyui-mixlab-nodes)     | ...                                                          |
| ComfyUI-CogVideoXWrapper    | kijai        | CogVideoX的Diffusers封装 - 模型：[CogVideo](https://github.com/THUDM/CogVideo)                                                                                                                                                                                                          | 1476  | 2025-02-17 00:48:16 | [GitHub](https://github.com/kijai/ComfyUI-CogVideoXWrapper)       | ...                                                          |
| ComfyUI-LTXVideo            | lightricks   | ComfyUI中LTX-Video支持的自定义节点。                                                                                                                                                                                                                                                      | 1036  | 2025-04-17 15:21:00 | [GitHub](https://github.com/Lightricks/ComfyUI-LTXVideo)          | ...                                                          |
| ComfyUI-VideoHelperSuite    | Kosinkadink  | 与视频工作流相关的节点。                                                                                                                                                                                                                                                                  | 939   | 2025-04-18 18:54:23 | [GitHub](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) | [详情](docs/zh/custom_nodes/videohelpersuite.md)             |
| VideoX-Fun                  | bubbliiiing  | VideoX-Fun是一个视频生成管线，可用于生成AI图像和视频，以及训练Diffusion Transformer的基线和Lora模型。                                                                                                                                                                                     | 914   | 2025-04-18 03:01:53 | [GitHub](https://github.com/aigc-apps/VideoX-Fun)                 | ...                                                          |
| Steerable Motion            | banodoco     | Steerable Motion是一个用于批量创意插值的ComfyUI节点。我们的目标是随着视频模型的发展，提供使用图像引导运动的最佳方法。                                                                                                                                                                         | 881   | 2024-06-15 23:01:54 | [GitHub](https://github.com/banodoco/steerable-motion)            | ...                                                          |
| ComfyUI-segment-anything-2  | kijai        | 使用 [segment-anything-2](https://github.com/facebookresearch/segment-anything-2) 进行图像或视频分割的节点。                                                                                                                                                                          | 867   | 2025-03-19 09:40:37 | [GitHub](https://github.com/kijai/ComfyUI-segment-anything-2)     | ...                                                          |

- 👉 [在 nodes.md (未来可能提供中文版 docs/zh/nodes_zh.md) 中查看所有节点包和详细信息](docs/nodes.md)

## 示例视频工作流
展示几个典型的节点组合工作流，侧重于实际应用场景。

- **工作流名称**
  - 使用案例
  -涉及节点
  - 简要流程说明
  - [可选：图表或链接]

---

> 每个部分建议只列出最常用或最具代表性的内容。更多详情请使用“查看更多”或链接，以避免正文过长。
