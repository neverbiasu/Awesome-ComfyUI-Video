# 📹 ComfyUI-VideoHelperSuite

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
| **CogVideo DualTextEncode**        | Process   | 同时对正向／负向提示进行编码，输出两组文本隐向量                       | ★★☆☆☆      | 需要同时传入正负提示并行控制时                             |
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



## 📑 核心节点详解

### (Down)load CogVideo Model

| 参数名        | 类型     | 默认值         | 参数性质 | 功能说明                                      |
| ------------- | -------- | -------------- | -------- | --------------------------------------------- |
| model_name    | STRING   | "cogvideo-1.1" | 可选     | 模型在 HuggingFace 上的名称                    |
| load_to_vram  | BOOLEAN  | True           | 可选     | 是否直接加载到显存                            |

**输出**  
- CogVideo 标准格式模型对象  

**使用场景**  
- 第一次运行或模型丢失时，从 HuggingFace 下载并加载标准 CogVideo 模型  

---

### (Down)load CogVideo GGUF Model

| 参数名        | 类型     | 默认值        | 参数性质 | 功能说明                                    |
| ------------- | -------- | ------------- | -------- | ------------------------------------------- |
| model_path    | STRING   | —             | 必选     | GGUF 格式模型本地路径或链接                 |
| quantized     | BOOLEAN  | True          | 可选     | 是否加载量化版本                            |

**输出**  
- GGUF 格式兼容模型，可用于低内存环境推理  

**使用场景**  
- 在支持 GGUF 的轻量化推理环境中加载模型，如 webUI 插件部署  

---

### (Down)load Tora Model

| 参数名        | 类型     | 默认值              | 参数性质 | 功能说明                                        |
| ------------- | -------- | ------------------- | -------- | ----------------------------------------------- |
| model_name    | STRING   | "cogvideox-tora"    | 可选     | Tora 格式优化模型在 HuggingFace 上的名称       |
| load_to_vram  | BOOLEAN  | True                | 可选     | 是否直接加载到显存                              |

**输出**  
- CogVideoX-Tora 加速模型对象  

**使用场景**  
- 使用阿里巴巴发布的 Tora 优化模型以获得推理性能提升  

---

### CogVideoX Model Loader

| 参数名        | 类型     | 默认值 | 参数性质 | 功能说明                                  |
| ------------- | -------- | ------ | -------- | ----------------------------------------- |
| path_or_name  | STRING   | —      | 必选     | 模型的本地路径或已缓存名称               |
| load_to_vram  | BOOLEAN  | True   | 可选     | 是否加载至 GPU                            |

**输出**  
- CogVideoX 模型对象，可用于 ComfyUI 后续流程  

**使用场景**  
- 管理与加载多个本地模型或实验版本  

---

### CogVideoX VAE Loader

| 参数名        | 类型     | 默认值     | 参数性质 | 功能说明                                  |
| ------------- | -------- | ---------- | -------- | ----------------------------------------- |
| vae_path      | STRING   | —          | 必选     | 指定的 VAE 权重路径                       |
| precision     | STRING   | "fp16"     | 可选     | 精度设置，支持 "fp16"、"bf16"、"fp32" 等  |

**输出**  
- VAE 解码器模块，用于与 CogVideoX 模型配套使用  

**使用场景**  
- 切换精度或实验不同图像解码器效果  

---

### (Down)load CogVideo ControlNet

| 参数名        | 类型     | 默认值           | 参数性质 | 功能说明                                         |
| ------------- | -------- | ---------------- | -------- | ------------------------------------------------ |
| control_type  | STRING   | "pose"           | 可选     | 控制类型，如 "pose"、"depth"、"canny" 等         |
| download_url  | STRING   | —                | 可选     | 自定义下载链接（如非默认模型源）                |

**输出**  
- ControlNet 模块，可作为条件引导器连接至采样器节点  

**使用场景**  
- 在视频生成中引入姿态、轮廓、深度图等结构约束，实现精细控制  

### CogVideo TextEncode

| 参数名        | 类型      | 默认值 | 参数性质 | 功能说明                                            |
| ------------- | --------- | ------ | -------- | --------------------------------------------------- |
| clip          | CLIP      | —      | 必选     | 提供 CLIP 模型实例用于文本编码，将提示文本转换为嵌入向量。  |
| prompt        | STRING    | “”     | 必选     | 输入的文本提示，用于指导视频生成的语义内容。  |
| strength      | FLOAT     | 1.0    | 可选     | 控制文本嵌入的强度，值越大提示影响越明显。  |
| force_offload | BOOLEAN   | False  | 可选     | 是否在 CPU 上加载模型以节省 GPU 显存。  |

**输出**  
- `CONDITIONING`：用于后续采样的文本条件嵌入。  
- `CLIP`：回传 CLIP 模型实例以便复用。   

**使用场景**  
- 将自然语言提示转换为模型可识别的条件向量，用于文本→视频或文本→图像流程。   

---

### CogVideo TextEncode Combine

| 参数名     | 类型     | 默认值 | 参数性质 | 功能说明                                             |
| ---------- | -------- | ------ | -------- | ---------------------------------------------------- |
| inputs     | LIST     | —      | 必选     | 多个来自 `CogVideo TextEncode` 或 `DualTextEncode` 的文本隐向量 |

**输出**  
- 合并后的文本 latent 表示，可用于统一传入 Sampler 节点中生成视频  

**使用场景**  
- 多重提示融合生成复杂场景内容，控制不同文本影响力

---

### CogVideo DualTextEncode

| 参数名        | 类型     | 默认值    | 参数性质 | 功能说明                                                         |
| ------------- | -------- | --------- | -------- | ---------------------------------------------------------------- |
| text          | STRING   | —         | 必选     | 主提示词，用于控制视频的主要生成内容                             |
| negative_text | STRING   | ""        | 可选     | 反向提示词，用于降低或排除某些不希望出现在生成视频中的内容        |
| model         | MODEL    | —         | 必选     | 已加载的 `CogVideo` 或兼容文本编码模型                            |
| return_dict   | BOOLEAN  | True      | 可选     | 是否以字典结构返回编码结果，False 则返回单一张量                 |

**输出**  
- TextEncoderOutput（当 return_dict=True）: 包含正向与负向文本的编码表示  
- torch.Tensor（当 return_dict=False）: 编码后文本张量，默认为正向文本 latent  

**使用场景**  
- 同时输入正向与反向提示词，实现更精细的生成控制，如“一个没有水印的动画风格场景”
- 搭配 Sampler 节点进行视频生成时的文本引导


### CogVideo Decode

| 参数名      | 类型     | 默认值 | 参数性质 | 功能说明                                              |
| ----------- | -------- | ------ | -------- | ----------------------------------------------------- |
| z           | LATENT   | —      | 必选     | 输入的潜在视频张量或嵌入，来自采样节点的输出。  |
| return_dict | BOOLEAN  | True   | 可选     | 是否以字典结构返回 `DecoderOutput`；否则仅返回 `torch.Tensor`。  |

**输出**  
- `DecoderOutput`（当 `return_dict=True`）: 包含解码后的视频张量及元数据。  
- `torch.Tensor`（当 `return_dict=False`）: 仅包含视频帧的张量表示。   

**使用场景**  
- 将模型推理得到的潜在表示解码为可视化的视频帧，用于保存或再次编码。   

---

### CogVideo Sampler

| 参数名             | 类型                | 默认值 | 参数性质 | 功能说明                                                |
| ------------------ | ------------------- | ------ | -------- | ------------------------------------------------------- |
| model              | COGVIDEOMODEL       | —      | 必选     | 要使用的 CogVideoX 模型实例，用于执行采样推理。  |
| positive           | CONDITIONING        | —      | 必选     | 正向条件嵌入，通常来自 `CogVideoTextEncode`。  |
| negative           | CONDITIONING        | —      | 可选     | 负向条件嵌入，用以去噪和对比控制。  |
| num_frames         | INT                 | —      | 必选     | 要生成的视频帧数。  |
| steps              | INT                 | —      | 可选     | 采样步数，影响质量与速度。  |
| cfg                | FLOAT               | —      | 可选     | 指导尺度，值越高对提示依赖越强。  |
| seed               | INT                 | —      | 可选     | 随机种子，用于结果复现。 : |
| scheduler          | ENUM                | —      | 可选     | 选择调度器算法，如 DDIM、DPM++、UniPC 等。  |
| samples            | LATENT              | —      | 可选     | 初始潜在张量，用于图像→视频或视频→视频流程。  |
| image_cond_latents | LATENT              | —      | 可选     | 图像编码后的潜在表示，用于图像到视频条件。  |
| denoise_strength   | FLOAT               | —      | 可选     | 去噪强度，用于视频→视频任务。  |
| controlnet         | COGVIDECONTROLNET   | —      | 可选     | ControlNet 分支输入，用于特定控制。  |
| tora_trajectory    | TORAFEATURES        | —      | 可选     | Tora 模型轨迹特征，用于高质量运动。  |
| fastercache        | FASTERCACHEARGS     | —      | 可选     | FasterCache 配置，用于内存/速度平衡。  |
| feta_args          | FETAARGS            | —      | 可选     | FreeNoise 噪声混洗参数。  |
| teacache_args      | TEACACHEARGS        | —      | 可选     | TeaCache 参数，用于重复推理优化。  |

**输出**  
- `LATENT`：包含生成的视频潜在表示，需通过 `CogVideo Decode` 解码。   

**使用场景**  
- 在图谱中执行基于文本、图像或已有视频的采样推理，生成视频潜在张量，用于文本→视频、图像→视频或视频样式迁移等。  

---

### CogVideo ImageEncode

| 参数名         | 类型     | 默认值   | 参数性质 | 功能说明                                                         |
| -------------- | -------- | -------- | -------- | ---------------------------------------------------------------- |
| image          | IMAGE    | —        | 必选     | 要编码的静态图像输入，可为单张或多张图像。  |
| processor      | VAE      | "auto"   | 可选     | 使用的 VAE 编码器类型，"auto" 则根据模型自动选择最佳 VAE。        |
| interpolation  | INT      | 1        | 可选     | 图像帧插值倍数，用于生成平滑过渡效果。                            |
| batch_size     | INT      | 1        | 可选     | 同时编码的图像数量，用于批量处理以提升效率。                      |

**输出**  
- `LATENT`：图像对应的潜在张量序列，可直接用于视频采样。   

**使用场景**  
- 将静态图片转换为视频管道中的潜在表示，支撑图像→视频或图像风格迁移流程。   

---

### CogVideo ImageEncode FunInP

| 参数名         | 类型       | 默认值   | 参数性质 | 功能说明                                                           |
| -------------- | ---------- | -------- | -------- | ------------------------------------------------------------------ |
| image          | IMAGE      | —        | 必选     | 要编码的静态图像输入，用于 Fun-In-P 模型流程。  |
| fun_model      | STRING     | "Fun-InP" | 必选     | 指定使用的 Fun-In-P 专用模型名称。                                  |
| interp_steps   | INT        | 1        | 可选     | 插值帧数，用于在 Fun-InP 编码时生成更多中间帧。                      |
| normalize      | BOOLEAN    | True     | 可选     | 是否对输入图像进行标准化处理，以提升编码稳定性。                      |

**输出**  
- `LATENT_FUN`：Fun-In-P 模型专用的潜在表示，优化了基于姿势或动画的编码质量。   

**使用场景**  
- 在 Fun-In-P（姿势驱动）视频生成或编辑流程中，将静态图像转换为具有动画潜力的潜在特征。   


| 参数名             | 类型   | 默认值     | 参数性质 | 功能说明                |
| ----------------- | ------ | ---------- | -------- | ----------------------- |
| video_path        | STRING | "input/"   | 必选     | 视频文件路径            |
| force_rate        | FLOAT  | 0.0        | 可选     | 强制调整帧率，设为0禁用 |
| force_size        | COMBO  | "Disabled" | 可选     | 快速调整尺寸选项        |
| frame_load_cap    | INT    | 0          | 可选     | 最大返回帧数(批次大小)  |
| skip_first_frames | INT    | 0          | 可选     | 跳过开头的帧数          |
| select_every_nth  | INT    | 1          | 可选     | 每N帧采样一帧           |

**输出**: IMAGE[] (图像序列), VHS_VIDEOINFO (视频信息), AUDIO (可选音频)

**使用场景**:
- 从服务器路径批量处理视频
- 处理网络URL视频
- 自动化工作流中使用



### Tora Encode Trajectory

| 参数名         | 类型      | 默认值 | 参数性质 | 功能说明                                             |
| -------------- | --------- | ------ | -------- | ---------------------------------------------------- |
| model          | TORA_MODEL| —      | 必选     | 传入已加载的 Tora 模型实例。                         |
| frames         | IMAGE[]   | —      | 必选     | 待处理的视频帧序列，用于提取运动轨迹特征。            |
| downsample     | INT       | 1      | 可选     | 对帧率或分辨率的下采样倍数，以加速特征提取。          |
| normalize      | BOOLEAN   | True   | 可选     | 是否对输入帧进行归一化处理，提升轨迹特征稳定性。      |

**输出**  
- `TORA_TRAJECTORY`：提取到的轨迹特征，用于后续采样或控制模块。  

**使用场景**  
- 在视频→视频或文本→视频生成管道中，为 CogVideoSampler 提供运动轨迹条件，使动画更连贯。  

---

### CogVideo LoraSelect

| 参数名         | 类型      | 默认值 | 参数性质 | 功能说明                                           |
| -------------- | --------- | ------ | -------- | -------------------------------------------------- |
| model          | COGVIDEOMODEL | —  | 必选     | 当前使用的基础 CogVideoX 模型实例。               |
| lora_name      | STRING    | —      | 必选     | 要加载的 LoRA 权重名称（如 “lora-style”）。       |
| lora_scale     | FLOAT     | 1.0    | 可选     | 应用于主模型的 LoRA 权重强度比例。                |
| merge          | BOOLEAN   | False  | 可选     | 是否将 LoRA 权重永久合并入基础模型。               |

**输出**  
- `MODIFIED_MODEL`：应用了 LoRA 权重的模型实例，可直接用于采样。  

**使用场景**  
- 在生成特定风格或细节强化的视频时，动态选择并应用 LoRA 权重而无需重启或重载主模型。  


### CogVideo ControlNet

| 参数名      | 类型                | 默认值 | 参数性质 | 功能说明                                               |
| ----------- | ------------------- | ------ | -------- | ------------------------------------------------------ |
| model       | COGVIDEOCONTROLNET  | —      | 必选     | 输入的 ControlNet 模型实例。  |
| conditioning| CONDITIONING        | —      | 必选     | 要施加的条件嵌入（如从 `CogVideoTextEncode` 或 `CogVideoImageEncode` 提供）。  |
| weight      | FLOAT               | 1.0    | 可选     | 控制 ControlNet 影响程度的权重比例。  |
| start_step  | INT                 | 0      | 可选     | 在采样步数中的起始应用步数。  |
| end_step    | INT                 | —      | 可选     | 结束应用的采样步数（默认到最后一步）。  |

**输出**  
- `MODIFIED_CONDITIONING`：应用 ControlNet 后的条件嵌入，可传入采样节点。  

**使用场景**  
- 在需要对生成过程施加结构化或时序约束时，将 ControlNet 与主模型配合使用。 

---

### CogVideoXFun ResizeToClosestBucket

| 参数名             | 类型        | 默认值 | 参数性质 | 功能说明                                                                                     |
| ------------------ | ----------- | ------ | -------- | -------------------------------------------------------------------------------------------- |
| `latents`          | LATENT      | —      | 必选     | 输入的视频潜在向量，可能与模型要求的分辨率或帧数不完全匹配。                                   |
| `bucket_heights`   | LIST[int]   | —      | 必选     | 支持的高度“bucket”列表，例如 `[256, 384, 512, 640]`。                                        |
| `bucket_widths`    | LIST[int]   | —      | 必选     | 支持的宽度“bucket”列表，需与 `bucket_heights` 对应索引一一匹配。                               |
| `num_frames`       | INT         | —      | 可选     | 可选提供帧数桶列表（与模型兼容的帧数），否则只调整空间维度。                                  |
| `mode`             | STRING      | `pad`  | 可选     | 空间调整模式：`pad`（零填充至桶大小），或 `crop`（中心裁剪至桶大小）。                         |
| `align_to_bucket`  | BOOLEAN     | True   | 可选     | 是否严格对齐到最近桶尺寸；若设置为 `False`，则取最接近但不超过当前尺寸的桶。                   |

**输出**  
- 返回 `ResizedLatents`：调整后与指定“bucket”尺寸对齐的视频潜在张量，形状为  
  `[batch, num_frames, channels, target_height, target_width]`。

**使用场景**  
- 在使用 Fun-InP 或其他对分辨率／帧数有严格“bucket”要求的模型前，对潜在向量进行预处理。  
- 配合 `CogVideoXFun Sampler`，确保输入尺寸合法且可被调度器正确处理，提高兼容性并避免报错。  

---

### CogVideo FasterCache

| 参数名            | 类型      | 默认值 | 参数性质 | 功能说明                                                             |
| ----------------- | --------- | ------ | -------- | -------------------------------------------------------------------- |
| `enable_cache`    | BOOLEAN   | True   | 可选     | 是否启用缓存机制，复用中间计算结果，减少反复计算。                     |
| `cache_size_mb`   | INT       | 512    | 可选     | 最大缓存大小（MB），超出后采用 LRU 策略清理最旧缓存。                 |
| `cache_dtype`     | STRING    | `"fp16"` | 可选   | 缓存张量的数据类型，可选 `"fp16"`、`"fp32"`，兼顾速度与精度。         |

**输出**  
- `cached`：布尔值，表示本次调用是否命中缓存。  
- `cache_info`：字典，包含当前缓存使用量与命中率等统计数据。  

**使用场景**  
- 对于反复调用相同条件下的较大模型推理，可通过缓存中间激活或权重编译结果，显著提升整体流水线速度。  

---

### CogVideo TorchCompileSettings

| 参数名               | 类型      | 默认值   | 参数性质 | 功能说明                                                             |
| -------------------- | --------- | -------- | -------- | -------------------------------------------------------------------- |
| `compile_mode`       | STRING    | `"default"` | 可选  | `torch.compile` 模式，可选 `"default"`、`"reduce-overhead"`、`"max-autotune"`，影响编译优化强度。 |
| `backend`            | STRING    | `"inductor"` | 可选 | 编译后端，可选 `"inductor"`、`"nvfuser"`，决定底层代码生成方式。       |
| `autotune`           | BOOLEAN   | False    | 可选     | 是否启用自动调优，尝试多种编译策略并选择最佳方案，启动时会额外耗时。     |

**输出**  
- `compile_settings`：字典，返回实际生效的编译模式、后端和自动调优状态。  

**使用场景**  
- 在 GPU 资源充足、对推理速度有极限需求的场景下，通过 PyTorch 2.0 编译加速模型执行；也可针对不同硬件切换不同后端。  

---

### CogVideo Context Options

| 参数名               | 类型       | 默认值       | 参数性质 | 功能说明                                                          |
| -------------------- | ---------- | ------------ | -------- | ----------------------------------------------------------------- |
| `context_window`     | INT        | 32           | 可选     | 最大时序上下文长度（帧数），超过则丢弃最早帧。                    |
| `free_noise_stride`  | INT        | 4            | 可选     | 在执行 free_noise 机制时，跳过多少帧作为扰动，控制噪声打乱颗粒度。 |
| `enable_pos_embed`   | BOOLEAN    | True         | 可选     | 是否为时序帧添加位置嵌入，改善顺序信息表示。                      |

**输出**  
- `context_config`：字典，包含最终上下文窗口、跳帧步长和位置嵌入状态。  

**使用场景**  
- 在长视频（Vid2Vid、Pose2Vid）生成中，需限制上下文长度以节省显存，同时在关键帧间加入扰动提升连贯性。  

---

### CogVideo LatentPreview

| 参数名             | 类型    | 默认值 | 参数性质 | 功能说明                                                         |
| ------------------ | ------- | ------ | -------- | ---------------------------------------------------------------- |
| `preview_frame`    | INT     | 0      | 可选     | 要可视化的帧索引（0 到 num_frames-1），在节点面板中展示该帧潜在图。 |
| `scale_factor`     | FLOAT   | 2.0    | 可选     | 将潜在图上采样以便在面板中更清晰预览。                            |

**输出**  
- `preview_image`：PIL Image 对象，在 UI 面板中显示对应帧的可视潜在图。  

**使用场景**  
- 调试中途生成阶段，可快速检查单帧潜在向量效果，帮助调整采样或模型参数。  

---

### CogVideo Enhance-A-Video

| 参数名             | 类型      | 默认值 | 参数性质 | 功能说明                                                                 |
| ------------------ | --------- | ------ | -------- | ------------------------------------------------------------------------ |
| `denoise_strength` | FLOAT     | 0.5    | 可选     | 后处理去噪强度（0.0–1.0），数值越高去噪越强但易损失细节。                  |
| `color_boost`      | FLOAT     | 1.2    | 可选     | 颜色增强倍数，用于对输出视频色彩进行微调，提高饱和度与对比度。             |
| `apply_style`      | STRING    | `None` | 可选     | 可选风格预设名称（如 `"cinematic"`、`"soft"`），对视频进行风格化处理。     |

**输出**  
- `enhanced_video`：后处理后的视频张量或文件路径，可直接保存或继续编码。  

**使用场景**  
- 在生成流程结束后，对视频进行去噪、调色或统一风格处理，以达到更具观赏性的最终效果。  

---

### CogVideo LoraSelect

| 参数名      | 类型    | 默认值 | 参数性质 | 功能说明                                     |
| ----------- | ------- | ------ | -------- | -------------------------------------------- |
| `model`     | MODEL   | —      | 必选     | 输入的 CogVideo 模型对象                     |
| `lora_path` | STRING  | —      | 必选     | 要加载的 LoRA 权重文件路径                   |
| `strength`  | FLOAT   | 1.0    | 可选     | LoRA 权重应用强度（0.0–1.0），控制效果强弱    |

**输出**  
- 应用了指定 LoRA 权重的 CogVideo 模型对象  

**使用场景**  
- 在视频生成中动态替换或叠加 LoRA 权重，实现风格微调或特效增强  

---

### CogVideo LoraSelect Comfy

| 参数名      | 类型    | 默认值 | 参数性质 | 功能说明                                          |
| ----------- | ------- | ------ | -------- | ------------------------------------------------- |
| `model`     | MODEL   | —      | 必选     | 输入的 CogVideo 模型对象                          |
| `lora_name` | STRING  | —      | 必选     | 预置 LoRA 名称（仓库或 ComfyUI 目录中已有权重）    |
| `strength`  | FLOAT   | 1.0    | 可选     | LoRA 权重应用强度（0.0–1.0），控制渲染效果比重     |

**输出**  
- 与 ComfyUI 原生系统兼容的、已加载 LoRA 权重的模型对象  

**使用场景**  
- 无缝调用 ComfyUI 的 LoRA 管理资源，快速切换不同预设权重  

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