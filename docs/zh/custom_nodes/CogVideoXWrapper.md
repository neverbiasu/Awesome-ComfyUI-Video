# 📹 ComfyUI-CogvideoXWrapper

## 🔍 集合概览

| 项目       | 详情                                                                  |
| ---------- | --------------------------------------------------------------------- |
| 📌 **作者** | [@Kkijai](https://github.com/kijai)                        |
| 📅 **版本** | 1.2.0+                                                                |
| 🏷️ **分类** | 模型开源拓展                                                              |
| 🔗 **仓库** | [GitHub链接](https://github.com/kijai/ComfyUI-CogVideoXWrapper) |

## 📝 功能简介
由 kijai 开发的一个开源扩展，它将最前沿的 CogVideoX 大规模文本—视频模型无缝接入到节点化的 ComfyUI 界面中，拥有模型下载、输入编码、采样推理到结果解码的一整套定制节点，实现“文字生成视频”、“图像生成视频”、“视频样式迁移”等多样化功能。

## 📊 节点一览表

| Node Name                          | Type      | Main Function                                                           | Complexity | Common Use Cases                                           |
| ---------------------------------- | --------- | ----------------------------------------------------------------------- | ---------- | ---------------------------------------------------------- |
| **(Down)load CogVideo Model**      | I/O       | 从 HuggingFace 下载并加载 CogVideo 标准格式模型                         | ★☆☆☆☆      | 第一次使用前获取模型                                        |
| **(Down)load CogVideo GGUF Model** | I/O       | 下载并加载 GGUF 格式的 CogVideo 模型                                    | ★☆☆☆☆      | 在仅支持 GGUF 的环境中加载模型                             |
| **(Down)load Tora Model**          | I/O       | 从 HuggingFace 下载并加载 CogVideoX Tora 格式加速模型                   | ★☆☆☆☆      | 使用阿里巴巴 Tora 优化版本进行加速推理                     |
| **CogVideoX Model Loader**         | I/O       | 加载指定路径或缓存中的 CogVideoX 模型                                   | ★☆☆☆☆      | 自定义模型管理与复用                                       |
| **CogVideoX VAE Loader**           | I/O       | 加载或切换 VAE 解码器                                                    | ★☆☆☆☆      | 在不同精度/格式间切换 VAE                                  |
| **(Down)load CogVideo ControlNet** | I/O       | 下载并加载 CogVideo ControlNet 条件引导模块                             | ★☆☆☆☆      | 对视频生成加入 ControlNet 引导；姿态／结构驱动生成         |
| **CogVideo ControlNet**            | I/O       | 应用 ControlNet 条件网络，将特定引导（如姿态、深度、边界框）注入生成流程 | ★★☆☆☆      | 姿态驱动、结构驱动、风格匹配等外部条件引导                  |
| **CogVideo TextEncode**            | Process   | 对单条文本提示进行编码，输出文本隐向量                                 | ★★☆☆☆      | 文本到视频流水线的前置编码                                  |
| **CogVideo TextEncode Combine**    | Process   | 将多路文本隐向量合并，为后续采样提供统一输入                            | ★★☆☆☆      | 混合多种提示语生成复杂情节                                  |
| **CogVideo ImageEncode**           | I/O       | 将静态图片编码为视频可用的时空潜在向量                                 | ★★★☆☆      | I2V 流程中的图像编码                                        |
| **CogVideo ImageEncode FunInP**    | I/O       | 为 Fun-InP （非官方 I2V）模型编码图像                                  | ★★★☆☆      | 使用 CogVideoX-Fun 图像到视频                              |
| **CogVideo Sampler**               | Process   | 基于文本／图像隐向量进行视频帧的扩散采样                               | ★★★★☆      | T2V 或 I2V 的核心采样                                       |
| **CogVideo Decode**                | I/O       | 将潜在向量解码回图像序列并输出为视频                                   | ★★☆☆☆      | 采样结束后生成最终视频输出                                 |
| **CogVideoXFun ResizeToClosestBucket** | Process | 将潜在向量自动调整到最近的采样“bucket”大小                          | ★☆☆☆☆      | 保证与 Fun 模型兼容的分辨率调整                            |
| **CogVideo FasterCache**        | Utility   | 启用 FasterCache 缓存优化，牺牲少量显存以换取更高推理速度                  | ★★☆☆☆      | 长视频或高分辨率场景下的速度提升                         |
| **CogVideo TorchCompileSettings** | Utility | 配置 `torch.compile` 优化选项                                             | ★★☆☆☆      | 使用 Triton/SageAttention 组合以获得编译加速             |
| **CogVideo Context Options**    | Utility   | 设置上下文窗口大小与 FreeNoise 噪声打乱策略                                 | ★★☆☆☆      | Vid2Vid 或 Pose2Vid 需要长序列上下文管理                |
| **CogVideo LatentPreview**      | Utility   | 在节点面板中预览中间潜在向量的效果                                         | ★☆☆☆☆      | 调试或可视化潜在空间生成效果                            |
| **CogVideo Enhance-A-Video**    | Process   | 对输出视频进行后期提亮、降噪或风格化处理                                     | ★★★☆☆      | 生成后增强，如色彩校正、去闪烁                           |
| **CogVideo LoraSelect**         | Advanced  | 在流水线中按名称或标签动态插入/切换 LoRA 权重                             | ★★☆☆☆      | 快速实验不同 LoRA 效果                                   |
| **CogVideo LoraSelect Comfy**   | Advanced  | 基于 ComfyUI 原生 LoRA 管理系统的无缝集成                                  | ★★★☆☆      | 与其它 ComfyUI LoRA 节点协同使用                       |
| **CogVideo TransformerEdit**    | Advanced  | 裁剪指定的 Transformer Block，移除不必要层以降低显存占用并提升推理效率      | ★★★★☆      | 模型轻量化；实验性层数对比；资源受限环境下的加速生成     |
| **Tora Encode Trajectory**      | Process   | 使用 Tora 的轨迹编码器，将用户绘制的运动路径转换为时空运动补丁隐向量     | ★★★★☆      | I2V 中精准控制运动轨迹；动画制作                            |

---

## 📑 核心节点详解

### (Down)load CogVideo Model

| 参数名                          | 类型                  | 默认值        | 参数性质 | 功能说明                                                                                  |
| ------------------------------- | --------------------- | ------------- | -------- | ----------------------------------------------------------------------------------------- |
| `model`                         | STRING                | —             | 必选     | 要下载并加载的 CogVideoX 模型标识，支持如 `THUDM/CogVideoX-2b`、`kijai/CogVideoX-5b-Tora` 等 |
| `block_edit`                    | TRANSFORMERBLOCKS     | —             | 可选     | 逗号分隔的 Transformer Block 索引列表，加载时会裁剪或替换指定层以调整模型容量与性能 |
| `lora`                          | COGLORA               | —             | 可选     | 指定要在加载时自动应用的 LoRA 权重（路径或名称），可用于风格微调或效果增强 |
| `compile_args`                  | COMPILEARGS           | —             | 可选     | 传递给 `torch.compile` 或 `diffusers` 编译接口的额外参数，用于控制后端优化行为 |
| `model.precision`               | ENUM (`fp16`,`fp32`,`bf16`) | `fp16`  | 可选     | 指定模型加载时的权重精度，较低精度可节省显存但可能略损质量               |
| `quantization`                  | ENUM                  | `disabled`    | 可选     | 量化后端选项，如 `fp8_e4m3fn`、`torchao_int8dq` 等，帮助在大模型上节省内存   |
| `enable_sequential_cpu_offload` | BOOLEAN               | `False`       | 可选     | 启用后分片地将模型子模块在 CPU/GPU 之间切换加载，以极大降低峰值显存占用    |
| `attention_mode`                | ENUM                  | `sdpa`        | 可选     | 选择注意力实现方式，如 `fused_sdpa`、`sageattn_qk_int8_pv_fp16_cuda`、`comfy` 等以优化速度/内存 |
| `load_device`                   | ENUM (`main_device`,`offload_device`) | `main_device` | 可选 | 控制模型组件的初始加载位置：`main_device`（GPU）或 `offload_device`（CPU）|

**输出**  
- `COGVIDEOMODEL`：加载完成并按上述参数配置的 `CogVideoXPipeline` 对象  
- `VAE`：与模型配套的 VAE 解码器模块  

**使用场景**  
- 在不同硬件环境（多 GPU / CPU + GPU）中自动优化模型加载方式  
- 结合 LoRA、量化、裁剪等手段平衡速度、显存与生成质量  
- 快速切换多种 CogVideoX 系列模型和版本以进行对比测试  

---

### (Down)load CogVideo GGUF Model

| 参数名                          | 类型         | 默认值   | 参数性质 | 功能说明                                                                                 |
| ------------------------------- | ------------ | -------- | -------- | ---------------------------------------------------------------------------------------- |
| `model`                         | STRING       | —        | 必选     | 要下载并加载的 CogVideo GGUF 模型名称                                                     |
| `vae_precision`                 | STRING       | `fp16`   | 可选     | VAE 组件的精度，可选 `bf16`、`fp16`、`fp32`                                               |
| `fp8_fastmode`                  | BOOLEAN      | `False`  | 可选     | 是否启用 FP8 快速模式，提升性能但可能略降精度                                             |
| `load_device`                   | STRING       | `cuda`   | 可选     | 模型加载设备，可选 `cpu` 或 `cuda`                                                       |
| `enable_sequential_cpu_offload` | BOOLEAN      | `False`  | 可选     | 是否启用顺序 CPU 溢出，按需将模型组件卸载到 CPU 以节省 GPU 显存                            |
| `attention_model`               | STRING       | `sdpa`   | 可选     | 注意力实现方式，可选 `sdpa`（标准）或 `sageattn`（SageAttention 加速，仅限 Linux）         |
| `block_edit`                    | LIST[int]    | —        | 可选     | 指定要修改或移除的 Transformer Block 索引列表，用于模型结构微调                           |

**输出**  
- `model`：加载并配置好的 CogVideo GGUF 模型对象  
- `vae`：相应精度的 VAE 解码器模块  

**使用场景**  
- 在低显存或移动端环境中，加载量化后的 GGUF 模型与匹配精度的 VAE，以平衡性能与资源占用。  

---

### (Down)load Tora Model

| 参数名         | 类型     | 默认值                      | 参数性质 | 功能说明                                                      |
| -------------- | -------- | --------------------------- | -------- | ------------------------------------------------------------- |
| `model_name`   | STRING   | `"kijai/CogVideoX-5b-Tora"` | 可选     | 要下载的 Tora 优化模型在 HuggingFace 上的仓库名称              |

**输出**  
- `TORAMODEL`：下载并实例化的 Tora 优化模型对象，可直接用于后续采样与解码节点  

**使用场景**  
- 在需要部署或测试 Tora‐优化的 CogVideoX 模型时，通过该节点一键下载并加载到显存，支持普通 T2V 或专用 I2V 版本。  

---

### CogVideoX Model Loader

| 参数名                          | 类型                    | 默认值          | 参数性质 | 功能说明                                                                                   |
| ------------------------------- | ----------------------- | --------------- | -------- | ------------------------------------------------------------------------------------------ |
| `model`                         | MODEL                   | —               | 必选     | 要加载的 CogVideoX 模型对象或标识（本地路径或缓存名称）                                      |
| `base_precision`                | ENUM(`fp16`,`fp32`,`bf16`) | `fp16`       | 可选     | 模型权重的基础精度，影响显存占用和数值范围                                                   |
| `quantization`                  | ENUM(...)               | `disabled`      | 可选     | 量化模式，支持多种 FP8/INT8/FP6 方案，用于进一步降低显存和加速推理                           |
| `enable_sequential_cpu_offload` | BOOLEAN                 | `False`         | 可选     | 是否启用顺序 CPU 卸载，将部分子模块权重从 GPU 转移到 CPU，以节省显存                         |
| `block_edit`                    | TRANSFORMERBLOCKS       | —               | 可选     | 指定要裁剪或重排的 Transformer Block 列表（由 `CogVideo TransformerEdit` 生成）             |
| `lora`                          | COGLORA                 | —               | 可选     | 要应用的 LoRA 权重配置（由 `CogVideo LoraSelect`/`Comfy` 节点提供）                         |
| `compile_args`                  | COMPILEARGS             | —               | 可选     | 传递给 `torch.compile` 的参数集，用于模型编译优化（如模式、后端、自动调优等）               |
| `attention_mode`                | ENUM(...)               | `sdpa`          | 可选     | 注意力计算模式，多种 SDPA/Sage/Fused 及 Comfy 自定义方案，影响速度与精度                     |

**输出**  
- `COGVIDEOMODEL`：加载并可进一步操作的 CogVideoX 模型实例  

**使用场景**  
- 在 ComfyUI 流水线开始阶段灵活加载不同格式或精度的 CogVideoX 模型  
- 配合 LoRA、Transformer 裁剪、torch.compile 等节点，按需定制模型性能与资源占用  

---

### CogVideoX VAE Loader

| 参数名       | 类型    | 默认值   | 参数性质 | 功能说明                                                    |
| ------------ | ------- | -------- | -------- | ----------------------------------------------------------- |
| `model_name` | STRING  | `"THUDM/CogVideoX-2b"` | 可选     | 要加载的 CogVideoX 模型名称或本地路径，支持 Diffusers 仓库格式    |
| `precision`  | STRING  | `"fp16"` | 可选     | 指定加载的 VAE 数据类型，可选 `"fp16"`、`"fp32"` 或 `"bf16"`      |

**输出**  
- `VAE`：加载后的 3D VAE 实例（`AutoencoderKLCogVideoX`），用于在解码时将潜在向量还原为视频帧。

**使用场景**  
- 在不同显存与性能需求下切换 VAE 精度（如低显存场景用 `fp16`，高保真场景用 `fp32` 或 `bf16`）。  
- 与主模型分离加载，便于复用或替换 VAE 解码器以测试不同解码效果。 

---

### (Down)load CogVideo ControlNet

| 参数名   | 类型       | 默认值 | 参数性质 | 功能说明                                                                                       |
| -------- | ---------- | ------ | -------- | ---------------------------------------------------------------------------------------------- |
| `model`  | COMBO[STRING] | —      | 必选     | 从下拉列表中选择要下载并加载的 CogVideoX ControlNet 模型名称|

**输出**  
- `COGVIDECONTROLNETMODEL`：已下载并加载的 ControlNet 模型对象，可作为后续 Apply ControlNet 节点的输入。  

**使用场景**  
- 在视频生成流程中注入姿态（HED）、边缘（Canny）等条件引导，实现对生成内容的精细控制。 

---

### CogVideo TextEncode

| 参数名        | 类型      | 默认值 | 参数性质 | 功能说明                                            |
| ------------- | --------- | ------ | -------- | --------------------------------------------------- |
| `clip`        | CLIP      | —      | 必选     | 提供 CLIP 模型实例用于文本编码，将提示文本转换为嵌入向量。  |
| `prompt`      | STRING    | `""`   | 必选     | 输入的文本提示，用于指导视频生成的语义内容。              |
| `strength`    | FLOAT     | `1.0`  | 可选     | 控制文本嵌入的强度，值越大提示影响越明显。                |
| `force_offload` | BOOLEAN | `False`| 可选     | 是否在 CPU 上加载模型以节省 GPU 显存。                    |

**输出**  
- `CONDITIONING`：用于后续采样的文本条件嵌入。  
- `CLIP`：回传 CLIP 模型实例以便复用。  

**使用场景**  
- 将自然语言提示转换为模型可识别的条件向量，用于文本→视频或文本→图像流程。  

---

### CogVideo Decode

| 参数名                   | 类型       | 默认值 | 参数性质 | 功能说明                                                   |
| ------------------------ | ---------- | ------ | -------- | ---------------------------------------------------------- |
| `vae`                    | VAE        | —      | 必选     | 用于解码的 VAE 模型                                        |
| `samples`                | LATENT     | —      | 必选     | 输入的潜在向量，来自采样节点的输出                          |
| `enable_vae_tiling`      | BOOLEAN    | —      | 可选     | 是否启用 VAE 平铺（tiling）解码，分块处理以降低显存压力      |
| `tile_sample_min_height` | INT        | —      | 可选     | 平铺解码时每块 tile 的最小高度，小于此值的 tile 将合并        |
| `tile_sample_min_width`  | INT        | —      | 可选     | 平铺解码时每块 tile 的最小宽度                              |
| `tile_overlap_factor_height` | FLOAT  | —      | 可选     | tile 在高度方向重叠比例，用于平滑边界                       |
| `tile_overlap_factor_width`  | FLOAT  | —      | 可选     | tile 在宽度方向重叠比例，用于平滑边界                       |
| `auto_tile_size`         | BOOLEAN    | —      | 可选     | 是否自动根据输入尺寸计算最优 tile 参数                      |

**输出**  
- `IMAGE`：解码后的图像序列，可直接用于视频合成或后续处理。  

**使用场景**  
- 将 CogVideoX Sampler 输出的潜在表示通过 VAE 解码为可视化视频帧，  
  同时可使用平铺模式处理大分辨率或长时序视频以节省显存并减少解码异常。 

---

### CogVideo TextEncode Combine

| 参数名                  | 类型       | 默认值             | 参数性质 | 功能说明                                                                                   |
| ----------------------- | ---------- | ------------------ | -------- | ------------------------------------------------------------------------------------------ |
| `conditioning_1`        | TENSOR     | —                  | 必选     | 第一个文本隐向量输入，通常来自 `CogVideo TextEncode` 或 `DualTextEncode` 节点的输出。      |
| `conditioning_2`        | TENSOR     | —                  | 必选     | 第二个文本隐向量输入，与 `conditioning_1` 形状相同，用于合并。                              |
| `combination_mode`      | STRING     | `"weighted_average"` | 可选   | 合并模式，可选：<br>• `"average"`：简单平均<br>• `"weighted_average"`：加权平均<br>• `"concatenate"`：沿最后维度拼接 |
| `weighted_average_ratio`| FLOAT      | `0.5`              | 可选     | 当 `combination_mode="weighted_average"` 时生效，取值范围 0.0–1.0，控制两个输入的权重比例。    |

**输出**  
- `conditioning`：合并后的文本隐向量（TENSOR），可直接传入采样器节点（如 `CogVideo Sampler`）进行视频生成。

**使用场景**  
- 将多个提示（如正向提示与反向提示，或不同主题提示）合并成一个统一的指导向量，灵活控制生成内容的风格与细节。  
- 通过 `"concatenate"` 模式保留各输入的完整特征，通过加权平均实现平滑过渡与平衡。  

---

### CogVideo Sampler

| 参数名                  | 类型               | 默认值            | 参数性质 | 功能说明                                                                                   |
| ----------------------- | ------------------ | ----------------- | -------- | ------------------------------------------------------------------------------------------ |
| `model`                 | MODEL              | —                 | 必选     | 已加载的 CogVideoX 模型实例，用于实际采样                                                   |
| `positive`              | TENSOR             | —                 | 必选     | 正向（positive）提示隐向量，用于指导采样                                                   |
| `negative`              | TENSOR             | —                 | 可选     | 负向（negative）提示隐向量，用于执行 classifier-free guidance                              |
| `samples`               | INT                | 1                 | 可选     | 每次调用生成的视频样本数量                                                                  |
| `images_cond_latent`    | LATENT             | —                 | 可选     | 图像条件编码隐向量（I2V 模式下），从 `CogVideo ImageEncode` 节点输出                         |
| `context_options`       | DICT               | —                 | 可选     | 由 `CogVideo Context Options` 节点生成的上下文配置                                          |
| `controlnet`            | LIST[MODULE, FLOAT]| []                | 可选     | 多路 ControlNet 模块及其强度（由 `(Down)load CogVideo ControlNet` 节点加载）                  |
| `tora_trajectory`       | TENSOR             | —                 | 可选     | `Tora Encode Trajectory` 节点输出的时空运动补丁隐向量                                       |
| `fastercache`           | DICT               | —                 | 可选     | `CogVideo FasterCache` 节点生成的缓存优化配置                                              |
| `feta_args`             | DICT               | —                 | 可选     | 传递给底层采样器的额外参数（如 fp8 模式等）                                                 |
| `num_frames`            | INT                | 16                | 可选     | 要生成的视频帧总数                                                                          |
| `steps`                 | INT                | 50                | 可选     | 扩散采样步数                                                                                |
| `cfg`                   | FLOAT              | 7.5               | 可选     | classifier-free guidance 强度                                                               |
| `seed`                  | INT                | 0                 | 可选     | 随机种子                                                                                   |
| `control_after_generate`| BOOLEAN            | False             | 可选     | 是否在完成潜在采样后再应用 ControlNet 条件                                                    |
| `scheduler`             | SCHEDULER          | PNDMScheduler     | 可选     | 噪声调度器实例                                                                              |
| `denoise_strength`      | FLOAT              | 1.0               | 可选     | 在 Vid2Vid/风格迁移等流程中控制去噪强度                                                      |

**输出**  
- `samples`：包含生成的视频潜在张量或解码后的视频帧，具体取决于后续解码节点配置

**使用场景**  
- 核心的 Text-to-Video、Image-to-Video、Video-to-Video 扩散采样节点，通过丰富的条件、上下文和优化选项生成高质量的视频样本  

---

### CogVideo ImageEncode

| 参数名              | 类型      | 默认值 | 参数性质 | 功能说明                                                                                  |
| ------------------- | --------- | ------ | -------- | ----------------------------------------------------------------------------------------- |
| `vae`               | VAE       | —      | 必选     | 指定用于编码的 Variational Autoencoder，用于将图像映射到时空潜在空间。                        |
| `start_image`       | IMAGE     | —      | 必选     | 起始帧图像，作为视频生成的第一个关键帧输入。                                                  |
| `end_image`         | IMAGE     | —      | 可选     | 结束帧图像，用于在 `start_image` 与 `end_image` 之间插值生成中间帧。                         |
| `enable_tiling`     | BOOLEAN   | False  | 可选     | 是否启用切片（tiling）编码，将图像分块处理以降低显存占用，适用于超大分辨率图像。              |
| `noise_aug_strength`| FLOAT     | 0.0    | 可选     | 对输入图像潜在向量添加噪声的强度，数值越高扰动越明显，可用于增加动态效果或抖动感。            |
| `strength`          | FLOAT     | 1.0    | 可选     | 控制原始图像特征在潜在表示中的保留比例，值越小越偏向随机噪声，值越大越保留原图细节。          |
| `start_percent`     | FLOAT     | 0.0    | 可选     | 插值时起始图像在混合中的占比（0.0–1.0），用于控制从 `start_image` 向 `end_image` 过渡的起始权重。 |
| `end_percent`       | FLOAT     | 1.0    | 可选     | 插值时结束图像在混合中的占比（0.0–1.0），用于控制过渡终点的权重。                            |

**输出**  
- `LATENT`：形状为 `[batch, num_frames, channels, height, width]` 的时空潜在表示，可直接输入到 `CogVideo Sampler` 或其他下游节点。

**使用场景**  
- **图像到视频（I2V）**：将静态图像或两张图像之间进行插帧，生成连贯的视频序列。  
- **动画制作**：结合 `start_image` 与 `end_image`，通过调整 `start_percent`/`end_percent` 实现帧间过渡动画。  
- **大尺寸编码**：在处理高分辨率图像时启用 `enable_tiling`，以降低显存峰值。    

---

### CogVideo ImageEncode FunInP

| 参数名               | 类型       | 默认值  | 参数性质 | 功能说明                                                                                       |
| -------------------- | ---------- | ------- | -------- | ---------------------------------------------------------------------------------------------- |
| `vae`                | VAE        | —       | 必选     | 用于编码的 VAE 解码器实例                                                                      |
| `start_image`        | IMAGE      | —       | 必选     | 起始帧图像，用作时空编码的首帧输入                                                              |
| `end_image`          | IMAGE      | —       | 必选     | 结束帧图像，用作时空编码的末帧输入                                                              |
| `num_frames`         | INT        | 16      | 可选     | 要生成的中间帧数量                                                                              |
| `enable_tiling`      | BOOLEAN    | False   | 可选     | 是否对输入图像进行平铺分块编码，以支持超高分辨率图像                                           |
| `noise_aug_strength` | FLOAT      | 0.0     | 可选     | 在编码过程中对图像添加噪声增强的强度，用于增加随机性或掩盖瑕疵                                 |

**输出**  
- `LATENT`：形状 `[batch, num_frames, ...]` 的时空潜在向量，可直接送入 Sampler 节点进行扩散采样。  

**使用场景**  
- Image-to-Video（I2V）流程：在文本引导或无文本场景下，将两帧静态图像及中间帧自动编码为视频潜在表示，适用于人物走动、物体平移等动画效果生成 。  
- 超高分辨率图像：启用平铺 (`enable_tiling=True`) 后可对大图分块编码，避免显存溢出，同时通过 `noise_aug_strength` 控制每块噪声一致性。  

---

### Tora Encode Trajectory

| 参数名           | 类型         | 默认值 | 参数性质 | 功能说明                                                         |
| ---------------- | ------------ | ------ | -------- | ---------------------------------------------------------------- |
| `tora_model`     | TORAMODEL    | —      | 必选     | 已加载的 Tora 模型，用于生成时空运动特征                         |
| `vae`            | VAE          | —      | 必选     | VAE 解码器模块，用于将运动补丁映射到潜在空间                      |
| `coordinates`    | STRING       | —      | 必选     | 用户定义的运动轨迹坐标（JSON/CSV 字符串），描述运动路径          |
| `width`          | INT          | —      | 可选     | 轨迹补丁的空间宽度（像素），应与原始图像宽度保持一致             |
| `height`         | INT          | —      | 可选     | 轨迹补丁的空间高度（像素），应与原始图像高度保持一致             |
| `num_frames`     | INT          | —      | 可选     | 要生成的轨迹补丁帧数                                             |
| `strength`       | FLOAT        | —      | 可选     | 轨迹编码强度，控制运动特征的影响比例                             |
| `start_percent`  | FLOAT        | —      | 可选     | 在采样过程中的开始注入百分比（0.0–1.0），决定何时开始叠加轨迹     |
| `end_percent`    | FLOAT        | —      | 可选     | 在采样过程中的停止注入百分比（0.0–1.0），决定何时停止叠加轨迹     |
| `enable_tiling`  | BOOLEAN      | —      | 可选     | 是否启用平铺分块处理，分批生成运动补丁以减少显存占用             |

**输出**  
- `TORAFEATURES`：编码后的时空运动特征张量，可直接输入至采样节点  
- `IMAGE`：运动轨迹可视化图，用于调试和预览轨迹分布  

**使用场景**  
- 在 I2V 流程中，将用户绘制的路径转换为 Tora 模型理解的运动补丁，精确控制生成视频中的主体运动轨迹。  

---

### CogVideo ControlNet

| 参数名                | 类型      | 默认值 | 参数性质 | 功能说明                                                                 |
| --------------------- | --------- | ------ | -------- | ------------------------------------------------------------------------ |
| `control_image`       | IMAGE     | —      | 必选     | 用于控制的视频帧条件图，如边缘（Canny）、HED、骨骼（Pose）、深度图等       |
| `controlnet_strength` | FLOAT     | 1.0    | 可选     | 控制信号强度，决定 ControlNet 条件对最终采样的影响比例（0.0–2.0）        |
| `start_percent`       | FLOAT     | 0.0    | 可选     | 控制影响开始在采样总步数中的相对位置（0.0–1.0），如 0.2 表示在 20% 步骤后 |
| `end_percent`         | FLOAT     | 1.0    | 可选     | 控制影响结束在采样总步数中的相对位置（0.0–1.0），如 0.8 表示在 80% 步骤前 |

**输出**  
- `controlnet_states`：处理后的条件潜在向量序列，可直接传入 CogVideo Sampler 进行融合采样  

**使用场景**  
- 在 Text-to-Video、Image-to-Video 或 Video-to-Video 流程中，引入结构、姿态、深度或其他可视化信息，精细化视频内容的布局和动作走向。  

---

### CogVideoXFun ResizeToClosestBucket

| 参数名             | 类型      | 默认值           | 参数性质 | 功能说明                                                                                           |
| ------------------ | --------- | ---------------- | -------- | -------------------------------------------------------------------------------------------------- |
| `images`           | IMAGE     | —                | 必选     | 输入的图像或潜在向量序列（帧），待调整分辨率以符合模型“桶”要求。                                      |
| `base_resolution`  | INT       | —                | 必选     | 与模型兼容的最小分辨率桶，例如 256、384、512；输出会对齐到此及其倍数。                                 |
| `upscale_method`   | STRING    | `nearest-exact`  | 可选     | 分辨率调整时使用的上采样方法：`nearest-exact`、`bilinear`、`area`、`bicubic` 或 `lanczos`。           |
| `crop`             | STRING    | `center`         | 可选     | 当原始分辨率高于目标桶时，裁剪模式：`disabled`（不裁剪）或 `center`（中心裁剪）。                     |

**输出**  
- `IMAGE`：按最近桶大小调整后的图像/潜在向量序列。  
- `INT`：调整后图像的宽度（像素）。  
- `INT`：调整后图像的高度（像素）。  

**使用场景**  
- 在使用 CogVideoX-Fun 或其他严格要求输入分辨率的模型前，自动将图像或潜在序列对齐至最近“桶”分辨率，避免尺寸不匹配错误，并可根据需求选择裁剪或上采样方法。 

---

### CogVideoX FasterCache

| 参数名                | 类型    | 默认值          | 参数性质 | 功能说明                                                                                     |
| --------------------- | ------- | --------------- | -------- | -------------------------------------------------------------------------------------------- |
| `start_step`          | INT     | 15              | 可选     | 从第几步开始启用缓存重用，跳过前面若干步的计算以节省显存和加速后续推理                          |
| `hf_step`             | INT     | —               | 可选     | 高频（high-frequency）特征缓存间隔：每隔多少步重用一次高频特征                                |
| `lf_step`             | INT     | —               | 可选     | 低频（low-frequency）特征缓存间隔：每隔多少步重用一次低频特征                                |
| `cache_device`        | STRING  | `"main_device"` | 可选     | 缓存存放设备，可选 `"main_device"`、`"offload_device"` 或 如 `"cuda:1"`                        |
| `num_blocks_to_cache` | INT     | —               | 可选     | 要缓存的 Transformer Block 数量，控制缓存粒度                                               |

**输出**  
- `FASTERCACHEARGS`：封装了所有缓存参数的配置对象，可直接传入采样器或模型推理函数中使用。

**使用场景**  
- 对长序列视频生成任务，通过跳过初始若干步并在中后期重用高/低频特征，显著降低重复计算，从而减少显存占用并提升整体推理速度。  

---

### CogVideo TorchCompileSettings

| 参数名                    | 类型      | 默认值        | 参数性质 | 功能说明                                                                                                     |
| ------------------------- | --------- | ------------- | -------- | ------------------------------------------------------------------------------------------------------------ |
| `backend`                | STRING    | `"inductor"`  | 可选     | 指定 `torch.compile` 使用的后端，可选包括 `"inductor"`（默认）、`"nvfuser"` 等自定义后端。                       |
| `mode`                   | STRING    | `"default"`   | 可选     | 编译模式，可选 `"default"`、`"reduce-overhead"`、`"max-autotune"` 等，控制优化强度与策略。                         |
| `fullgraph`              | BOOLEAN   | `False`       | 可选     | 是否启用 full-graph 模式；`True` 时捕获整个模型为单个图，否则在图破裂时可能报错。                              |
| `dynamic`                | BOOLEAN   | `False`       | 可选     | 是否开启动态形状支持，生成更通用的内核以减少因输入尺寸变化导致的二次编译。                                      |
| `dynamo_cache_size_limit`| INT       | `8`           | 可选     | 设置 `torch._dynamo.config.cache_size_limit`（默认 8），控制单个函数可生成的最大编译缓存版本数，防止无限编译。      |

**输出**  
- `torch_compile_args`：封装了实际生效的编译参数，包括 `backend`、`mode`、`fullgraph`、`dynamic` 及 `dynamo_cache_size_limit`。

**使用场景**  
- 在 GPU 资源充足且对推理速度有苛刻要求的场景下，通过 PyTorch 2.0+ 的 `torch.compile` 显著加速模型执行。  
- 根据不同硬件特性和模型结构，灵活切换后端与模式以获得最佳性能–稳定性平衡。  
- 调试时可通过调整 `dynamic` 与 `fullgraph` 参数，定位和解决编译失败或性能瓶颈。  

---

### CogVideo Context Options

| 参数名              | 类型       | 默认值                 | 参数性质 | 功能说明                                                                 |
| ------------------- | ---------- | ---------------------- | -------- | ------------------------------------------------------------------------ |
| `context_schedule`  | STRING     | `"uniform_standard"`   | 可选     | 上下文调度策略，可选：<br>• `uniform_standard`（均匀标准）<br>• `uniform_looped`（循环均匀）<br>• `static_standard`（静态标准） |
| `context_frames`    | INT        | `32`                   | 可选     | 最大保留上下文帧数，当输入帧数超过该值时，最早帧会被丢弃                  |
| `context_stride`    | INT        | `1`                    | 可选     | 从原始序列抽取上下文帧的步长，控制帧间隔                                   |
| `context_overlap`   | INT        | `0`                    | 可选     | 相邻上下文窗口之间的重叠帧数，用于平滑过渡                                 |
| `freenoise`         | BOOLEAN    | `False`                | 可选     | 是否启用 FreeNoise 噪声打乱机制，在上下文帧中定期加入随机扰动               |

**输出**  
- `COGCONTEXT`：字典，包含以上所有生效的上下文配置参数。

**使用场景**  
- 在 Vid2Vid、Pose2Vid 等长序列视频生成中，通过限制保留帧数、设置步长与重叠，以及可选的噪声扰动，平衡显存占用与生成连贯性。  
- 针对不同场景灵活调整调度策略（如循环 vs 静态）和噪声机制，优化动作连贯度或增加画面多样性。  

---

### CogVideo LatentPreview

| 参数名      | 类型     | 默认值 | 参数性质 | 功能说明                                                    |
| ----------- | -------- | ------ | -------- | ----------------------------------------------------------- |
| `samples`   | LATENT   | —      | 必选     | 输入的视频潜在张量，形状 `[batch, num_frames, channels, H, W]` |
| `seed`      | INT      | —      | 可选     | 用于生成预览的随机种子，保证可重复性                         |
| `min_val`   | FLOAT    | —      | 可选     | 可视化时映射的最小潜在值，低于此值的部分会被裁剪             |
| `max_val`   | FLOAT    | —      | 可选     | 可视化时映射的最大潜在值，高于此值的部分会被裁剪             |
| `r_bias`    | FLOAT    | —      | 可选     | 红色通道偏移，用于调节预览图中的红色分量                     |
| `g_bias`    | FLOAT    | —      | 可选     | 绿色通道偏移，用于调节预览图中的绿色分量                     |
| `b_bias`    | FLOAT    | —      | 可选     | 蓝色通道偏移，用于调节预览图中的蓝色分量                     |

**输出**  
- `IMAGE`：预览图像，展示指定潜在帧的可视化效果。  
- `STRING`：文本信息，包含当前 `seed`、`min_val`/`max_val` 范围及各通道偏移量等参数详情。

**使用场景**  
- 在视频生成流程中实时可视化中间潜在表示，帮助调试、校准映射范围及颜色偏移参数，以便快速定位和优化生成效果。


---

### CogVideo Enhance-A-Video

| 参数名         | 类型    | 默认值 | 参数性质 | 功能说明                                                          |
| -------------- | ------- | ------ | -------- | ----------------------------------------------------------------- |
| `weight`       | FLOAT   | 1.0    | 可选     | 增强温度因子，乘以跨帧注意力强度，用于提高视频连贯性和细节表现。    |
| `start_percent`| FLOAT   | 0.0    | 可选     | 从视频起始位置开始应用增强的百分比（0.0–1.0）。                   |
| `end_percent`  | FLOAT   | 1.0    | 可选     | 到视频结束位置停止增强的百分比（0.0–1.0）。                       |

**输出**  
- `FETAARGS`：包含增强后跨帧注意力调整参数的数据结构，可在后续解码或渲染环节使用。 

**使用场景**  
- 在生成流程结束后，对视频的时序注意力输出进行无训练微调，提升视频细节和画面连贯性，尤其适合人物运动或场景切换频繁的视频。  

### CogVideo LoraSelect

| 参数名       | 类型    | 默认值 | 参数性质 | 功能说明                                         |
| ------------ | ------- | ------ | -------- | ------------------------------------------------ |
| `model`      | MODEL   | —      | 必选     | 输入的 CogVideo 模型或流水线对象                 |
| `lora_path`  | STRING  | —      | 必选     | 本地文件系统或远程 URL 上的 LoRA 权重文件路径     |
| `lora_scale` | FLOAT   | 1.0    | 可选     | LoRA 权重应用强度（0.0–1.0），控制效果强弱        |
| `unet_only`  | BOOLEAN | False  | 可选     | 是否仅将 LoRA 应用于 UNet 子模块                  |

**输出**  
- `model`：已加载并应用了指定 LoRA 权重的模型对象  

**使用场景**  
- 在视频生成过程中动态引入自定义 LoRA 权重，实现风格微调或特殊效果增强。

---

### CogVideo LoraSelect Comfy

| 参数名       | 类型    | 默认值 | 参数性质 | 功能说明                                                    |
| ------------ | ------- | ------ | -------- | ----------------------------------------------------------- |
| `model`      | MODEL   | —      | 必选     | 输入的 CogVideo 模型或流水线对象                            |
| `lora_name`  | STRING  | —      | 必选     | 存放在 ComfyUI 默认 LoRA 目录中的预置 LoRA 权重名称         |
| `strength`   | FLOAT   | 1.0    | 可选     | LoRA 权重应用强度（0.0–1.0），控制渲染效果比重              |
| `overwrite`  | BOOLEAN | False  | 可选     | 是否覆盖模型中已有所有 LoRA delta（`True` 覆盖，`False` 叠加） |

**输出**  
- `model`：已加载并应用 ComfyUI 预置 LoRA 权重的模型对象  

**使用场景**  
- 利用 ComfyUI 原生 LoRA 管理系统，无需手动指定路径即可快速切换并应用预设权重。

---

### CogVideo TransformerEdit

| 参数名           | 类型     | 默认值 | 参数性质 | 功能说明                                              |
| ---------------- | -------- | ------ | -------- | ----------------------------------------------------- |
| `remove_blocks`  | STRING   | “”     | 必选     | 逗号分隔的 Transformer Block 索引列表，如 `"15,25,37"` |

**输出**  
- `block_list`：整型列表，实际移除的 Block 索引  

**使用场景**  
- 精准裁剪模型层数，减小显存占用或加速推理；用于实验性层数对比  

---

### Tora Encode Trajectory

| 参数名       | 类型      | 默认值     | 参数性质 | 功能说明                                           |
| ------------ | --------- | ---------- | -------- | -------------------------------------------------- |
| `trajectory` | PATH      | —          | 必选     | 用户绘制或导入的运动轨迹文件（SVG、JSON 等）       |
| `resolution` | STRING    | `"512x512"`| 可选     | 生成潜在补丁的空间分辨率（宽×高）                  |
| `frame_count`| INTEGER   | 16         | 可选     | 要生成的运动补丁帧数                                |

**输出**  
- `trajectory_embeds`：时空运动补丁隐向量，可直接用于采样节点  

**使用场景**  
- 在 I2V 生成流程中按自定义轨迹控制运动；适用于动画制作与路径驱动特效  

## 🔧 常用工作流组合

| 工作流名称               | 节点组合                                      | 用途                                                                 |
|--------------------------|---------------------------------------------|----------------------------------------------------------------------|
| 文本到视频生成           | CogVideo TextEncode → CogVideo Sampler → CogVideo Decode | 根据文本提示生成视频内容（核心功能，稳定性最高）                     |
| 图像到视频转换           | CogVideo ImageEncode → CogVideo Sampler → CogVideo Decode | 将静态图像扩展为动态视频（需验证图像编码器兼容性）                   |
| 控制运动轨迹的视频生成   | Tora Encode Trajectory → CogVideo Sampler → CogVideo Decode | 通过轨迹坐标控制物体运动路径（依赖轨迹编码器精度）                   |
| 加速视频生成             | CogVideo FasterCache → CogVideo Sampler → CogVideo Decode | 通过显存优化提升生成速度（实测提速20%-30%）                          |
| 上下文窗口调整           | CogVideo Context Options → CogVideo Sampler → CogVideo Decode | 调整时序上下文长度（16-64帧范围有效）                                |
| 视频增强处理             | CogVideo Decode → CogVideo Enhance-A-Video  | 分辨率提升/插帧增强（需单独部署增强模块）                            |

---