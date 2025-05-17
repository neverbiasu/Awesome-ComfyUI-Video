# 🎥 EasyAnimate

## 🔍 集合概览

| 项目       | 详情                                                                 |
| ---------- | -------------------------------------------------------------------- |
| 📌 **作者** | [@aigc-apps](https://github.com/aigc-apps/EasyAnimate)               |
| 📅 **版本** | 5.1+                                                                |
| 🏷️ **分类** | 视频生成与控制                                                      |
| 🔗 **仓库** | [GitHub链接](https://github.com/aigc-apps/EasyAnimate)               |

## 📝 功能简介
EasyAnimate 是一个强大的视频生成工具，支持从文本、图像或视频生成高质量视频，并提供多种控制功能（如相机轨迹、深度、姿态等）。

## 📊 节点一览表

| 节点名称                        | 类型          | 主要功能                                   | 复杂度 | 常用场景                     |
| ------------------------------- | ------------- | ------------------------------------------ | ------ | ---------------------------- |
| **Load EasyAnimate Model**      | 模型加载      | 加载 EasyAnimate 模型                      | ★★☆☆☆  | 初始化模型                   |
| **Load EasyAnimate Lora**       | 模型加载      | 加载 EasyAnimate 的 Lora 权重              | ★★☆☆☆  | 模型微调                     |
| **EasyAnimate_TextBox**         | 输入          | 提供文本输入框                             | ★☆☆☆☆  | 输入提示词                   |
| **EasyAnimate Sampler for T2V** | 采样器        | 文本到视频采样                             | ★★★☆☆  | 文本到视频生成               |
| **EasyAnimateV5 Sampler for T2V**| 采样器        | 文本到视频采样 (V5+)                       | ★★★☆☆  | 文本到视频生成 (V5+)         |
| **EasyAnimate Sampler for I2V** | 采样器        | 图像到视频采样                             | ★★★☆☆  | 图像到视频生成               |
| **EasyAnimateV5 Sampler for I2V**| 采样器        | 图像到视频采样 (V5+)                       | ★★★☆☆  | 图像到视频生成 (V5+)         |
| **EasyAnimate Sampler for V2V** | 采样器        | 视频到视频采样                             | ★★★☆☆  | 视频风格迁移                 |
| **EasyAnimateV5 Sampler for V2V**| 采样器        | 视频到视频采样 (V5+)                       | ★★★☆☆  | 视频风格迁移 (V5+)         |
| **Create Trajectory Based On KJNodes** | 工具   | 基于 KJNodes 创建轨迹                      | ★★☆☆☆  | 相机轨迹控制                 |
| **CameraBasicFromChaoJie**      | 工具          | 基础相机控制                               | ★★☆☆☆  | 简单相机运动                 |
| **CameraTrajectoryFromChaoJie**| 工具          | 生成相机轨迹参数                           | ★★☆☆☆  | 高级相机运动                 |
| **CameraCombineFromChaoJie**    | 工具          | 合并多个相机运动轨迹                       | ★★☆☆☆  | 复杂相机运动                 |
| **CameraJoinFromChaoJie**       | 工具          | 连接多个相机轨迹                           | ★★☆☆☆  | 轨迹拼接                     |
| **ImageMaximumNode**            | 工具          | 合并两段视频帧                             | ★★☆☆☆  | 视频帧合并                   |

## 📑 核心节点详解

### Load EasyAnimate Model

**描述：**
加载 EasyAnimate 模型，支持多种模型类型和精度。

**输入参数表：**

| 参数名           | 类型   | 默认值                   | 参数性质 | 功能说明                     |
| ---------------- | ------ | ------------------------ | -------- | ---------------------------- |
| model            | STRING | EasyAnimateV5.1-12b-zh  | 必选     | 要加载的模型名称             |
| GPU_memory_mode  | STRING | model_cpu_offload       | 可选     | GPU 内存模式                 |
| model_type       | STRING | Inpaint                | 可选     | 模型类型（如 Inpaint 或 Control） |
| config           | STRING | easyanimate_video_v5.1_magvit_qwen.yaml | 必选 | 模型配置文件路径             |
| precision        | STRING | bf16                   | 可选     | 模型精度（如 fp16 或 bf16）  |

**输出：**

* `EASYANIMATESMODEL`: 加载的 EasyAnimate 模型对象。

**使用场景：**
- 初始化 EasyAnimate 模型以进行视频生成。

---

### Load EasyAnimate Lora

**描述：**
加载 EasyAnimate 的 Lora 权重，用于模型微调。

**输入参数表：**

| 参数名           | 类型   | 默认值 | 参数性质 | 功能说明                     |
| ---------------- | ------ | ------ | -------- | ---------------------------- |
| easyanimate_model | EASYANIMATESMODEL | -      | 必选     | 加载的 EasyAnimate 模型对象 |
| lora_name        | STRING | -      | 必选     | Lora 权重文件名              |
| strength_model   | FLOAT  | 1.0    | 可选     | Lora 权重强度                |
| lora_cache       | BOOLEAN| False  | 可选     | 是否启用 Lora 缓存           |

**输出：**

* `EASYANIMATESMODEL`: 加载了 Lora 权重的模型对象。

**使用场景：**
- 对模型进行微调以适应特定任务。

---

### EasyAnimate_TextBox

**描述：**
提供一个多行文本输入框，通常用于输入提示词。

**输入参数表：**

| 参数名   | 类型   | 默认值 | 参数性质 | 功能说明   |
| -------- | ------ | ------ | -------- | ---------- |
| prompt   | STRING | ""     | 必选     | 文本输入内容 |

**输出：**

* `prompt` (STRING_PROMPT): 输出的文本内容，可连接到采样器节点的提示词输入。

**使用场景：**
- 为EasyAnimate采样器提供正面或负面提示词。

---

### EasyAnimate Sampler for T2V

**描述：**
从文本生成视频。

**输入参数表：**

| 参数名           | 类型   | 默认值 | 参数性质 | 功能说明                     |
| ---------------- | ------ | ------ | -------- | ---------------------------- |
| easyanimate_model | EASYANIMATESMODEL | -      | 必选     | 加载的 EasyAnimate 模型对象 |
| prompt           | STRING | -      | 必选     | 文本提示                     |
| negative_prompt  | STRING | -      | 可选     | 负面提示                     |
| video_length     | INT    | 72     | 必选     | 视频帧数                     |
| width            | INT    | 1008   | 必选     | 视频宽度                     |
| height           | INT    | 576    | 必选     | 视频高度                     |
| is_image         | BOOLEAN| False  | 可选     | 是否生成单张图片             |
| seed             | INT    | 43     | 可选     | 随机种子                     |
| steps            | INT    | 25     | 必选     | 采样步数                     |
| cfg              | FLOAT  | 7.0    | 必选     | 引导强度                     |
| scheduler        | STRING | DDIM   | 必选     | 采样器类型 (可选: Euler, Euler A, DPM++, PNDM, DDIM) |

**输出：**

* `IMAGE`: 生成的视频帧序列。

**使用场景：**
- 根据文本描述生成视频内容。

---

### EasyAnimateV5 Sampler for T2V

**描述：**
从文本生成视频（EasyAnimate V5+ 版本优化）。此版本通常包含针对V5系列模型的特定优化和参数，如 `teacache`。

**输入参数表：**

| 参数名             | 类型   | 默认值 | 参数性质 | 功能说明                     |
| ------------------ | ------ | ------ | -------- | ---------------------------- |
| easyanimate_model  | EASYANIMATESMODEL | -      | 必选     | 加载的 EasyAnimate 模型对象 |
| prompt             | STRING_PROMPT | -      | 必选     | 文本提示                     |
| negative_prompt    | STRING_PROMPT | -      | 可选     | 负面提示                     |
| video_length       | INT    | 49     | 必选     | 视频帧数 (V5通常为1-49)      |
| width              | INT    | 1008   | 必选     | 视频宽度                     |
| height             | INT    | 576    | 必选     | 视频高度                     |
| is_image           | BOOLEAN| False  | 可选     | 是否生成单张图片             |
| seed               | INT    | 43     | 可选     | 随机种子                     |
| steps              | INT    | 25     | 必选     | 采样步数                     |
| cfg                | FLOAT  | 7.0    | 必选     | 引导强度                     |
| scheduler          | COMBO  | Flow   | 必选     | 采样器类型 (V5推荐Flow)      |
| teacache_threshold | FLOAT  | 0.10   | 可选     | TeaCache阈值                 |
| enable_teacache    | BOOLEAN| True   | 可选     | 是否启用TeaCache             |

**输出：**

* `IMAGE`: 生成的视频帧序列。

**使用场景：**
- 使用EasyAnimate V5及更高版本模型进行文本到视频生成，利用其特定优化。

---

### EasyAnimate Sampler for I2V

**描述：**
从图像生成视频。

**输入参数表：**

| 参数名           | 类型   | 默认值 | 参数性质 | 功能说明                     |
| ---------------- | ------ | ------ | -------- | ---------------------------- |
| easyanimate_model | EASYANIMATESMODEL | -      | 必选     | 加载的 EasyAnimate 模型对象 |
| prompt           | STRING | -      | 必选     | 文本提示                     |
| negative_prompt  | STRING | -      | 可选     | 负面提示                     |
| video_length     | INT    | 72     | 必选     | 视频帧数                     |
| base_resolution  | COMBO  | 768    | 必选     | 基础分辨率 (512, 768, 960, 1024) |
| seed             | INT    | 43     | 可选     | 随机种子                     |
| steps            | INT    | 25     | 必选     | 采样步数                     |
| cfg              | FLOAT  | 7.0    | 必选     | 引导强度                     |
| scheduler        | STRING | DDIM   | 必选     | 采样器类型 (可选: Euler, Euler A, DPM++, PNDM, DDIM) |
| start_img        | IMAGE  | -      | 可选     | 起始图像                     |
| end_img          | IMAGE  | -      | 可选     | 结束图像                     |

**输出：**

* `IMAGE`: 生成的视频帧序列。

**使用场景：**
- 从图像生成连续的视频内容。

---

### EasyAnimateV5 Sampler for I2V

**描述：**
从图像生成视频（EasyAnimate V5+ 版本优化）。此版本通常包含针对V5系列模型的特定优化和参数，如 `teacache`。

**输入参数表：**

| 参数名             | 类型   | 默认值 | 参数性质 | 功能说明                     |
| ------------------ | ------ | ------ | -------- | ---------------------------- |
| easyanimate_model  | EASYANIMATESMODEL | -      | 必选     | 加载的 EasyAnimate 模型对象 |
| prompt             | STRING_PROMPT | -      | 必选     | 文本提示                     |
| negative_prompt    | STRING_PROMPT | -      | 可选     | 负面提示                     |
| video_length       | INT    | 49     | 必选     | 视频帧数 (V5通常为1-49)      |
| base_resolution    | COMBO  | 768    | 必选     | 基础分辨率 (512, 768, 960, 1024) |
| seed               | INT    | 43     | 可选     | 随机种子                     |
| steps              | INT    | 25     | 必选     | 采样步数                     |
| cfg                | FLOAT  | 7.0    | 必选     | 引导强度                     |
| scheduler          | COMBO  | Flow   | 必选     | 采样器类型 (V5推荐Flow)      |
| teacache_threshold | FLOAT  | 0.10   | 可选     | TeaCache阈值                 |
| enable_teacache    | BOOLEAN| True   | 可选     | 是否启用TeaCache             |
| start_img          | IMAGE  | -      | 可选     | 起始图像                     |
| end_img            | IMAGE  | -      | 可选     | 结束图像                     |

**输出：**

* `IMAGE`: 生成的视频帧序列。

**使用场景：**
- 使用EasyAnimate V5及更高版本模型进行图像到视频生成，利用其特定优化。

---

### EasyAnimate Sampler for V2V

**描述：**
从视频生成视频。

**输入参数表：**

| 参数名           | 类型   | 默认值 | 参数性质 | 功能说明                     |
| ---------------- | ------ | ------ | -------- | ---------------------------- |
| easyanimate_model | EASYANIMATESMODEL | -      | 必选     | 加载的 EasyAnimate 模型对象 |
| prompt           | STRING | -      | 必选     | 文本提示                     |
| negative_prompt  | STRING | -      | 可选     | 负面提示                     |
| video_length     | INT    | 72     | 必选     | 视频帧数                     |
| base_resolution  | COMBO  | 768    | 必选     | 基础分辨率 (512, 768, 960, 1024) |
| seed             | INT    | 43     | 可选     | 随机种子                     |
| steps            | INT    | 25     | 必选     | 采样步数                     |
| cfg              | FLOAT  | 7.0    | 必选     | 引导强度                     |
| denoise_strength | FLOAT  | 0.70   | 必选     | 重绘强度                     |
| scheduler        | STRING | DDIM   | 必选     | 采样器类型 (可选: Euler, Euler A, DPM++, PNDM, DDIM) |
| validation_video | IMAGE  | -      | 可选     | 输入验证视频                 |
| control_video    | IMAGE  | -      | 可选     | 输入控制视频                 |

**输出：**

* `IMAGE`: 生成的视频帧序列。

**使用场景：**
- 对输入视频进行风格迁移或内容修改。

---

### EasyAnimateV5 Sampler for V2V

**描述：**
从视频生成视频（EasyAnimate V5+ 版本优化）。此版本通常包含针对V5系列模型的特定优化和参数，如 `teacache`，并支持更复杂的控制，如相机条件。

**输入参数表：**

| 参数名             | 类型   | 默认值 | 参数性质 | 功能说明                     |
| ------------------ | ------ | ------ | -------- | ---------------------------- |
| easyanimate_model  | EASYANIMATESMODEL | -      | 必选     | 加载的 EasyAnimate 模型对象 |
| prompt             | STRING_PROMPT | -      | 必选     | 文本提示                     |
| negative_prompt    | STRING_PROMPT | -      | 可选     | 负面提示                     |
| video_length       | INT    | 49     | 必选     | 视频帧数 (V5通常为1-49)      |
| base_resolution    | COMBO  | 768    | 必选     | 基础分辨率 (512, 768, 960, 1024) |
| seed               | INT    | 43     | 可选     | 随机种子                     |
| steps              | INT    | 25     | 必选     | 采样步数                     |
| cfg                | FLOAT  | 7.0    | 必选     | 引导强度                     |
| denoise_strength   | FLOAT  | 0.70   | 必选     | 重绘强度                     |
| scheduler          | COMBO  | Flow   | 必选     | 采样器类型 (V5推荐Flow)      |
| teacache_threshold | FLOAT  | 0.10   | 可选     | TeaCache阈值                 |
| enable_teacache    | BOOLEAN| True   | 可选     | 是否启用TeaCache             |
| validation_video   | IMAGE  | -      | 可选     | 输入验证视频 (用于Inpaint模型) |
| control_video      | IMAGE  | -      | 可选     | 输入控制视频 (用于Control模型) |
| camera_conditions  | STRING | -      | 可选     | 相机运动条件 (JSON字符串)    |
| ref_image          | IMAGE  | -      | 可选     | 参考图像                     |

**输出：**

* `IMAGE`: 生成的视频帧序列。

**使用场景：**
- 使用EasyAnimate V5及更高版本模型进行视频到视频的风格迁移或内容修改，支持相机控制。

---

### Create Trajectory Based On KJNodes

**描述：**
基于 KJNodes 的坐标和遮罩输入创建轨迹图像，用于EasyAnimate的轨迹控制。轨迹点会以高斯热图的形式绘制在图像上。

**输入参数表：**

| 参数名           | 类型   | 默认值 | 参数性质 | 功能说明                     |
| ---------------- | ------ | ------ | -------- | ---------------------------- |
| coordinates      | STRING | -      | 必选     | 轨迹坐标 (JSON格式字符串列表或单个JSON字符串) |
| masks            | MASK   | -      | 必选     | 轨迹遮罩 (用于确定输出图像尺寸) |

**输出：**

* `IMAGE`: 生成的轨迹图像序列。

**使用场景：**
- 为EasyAnimate的Control模型提供轨迹控制图。
- 结合KJNodes的曲线工具定义复杂的运动路径。

---

### CameraBasicFromChaoJie

**描述：**
生成基础相机运动。

**输入参数表：**

| 参数名           | 类型   | 默认值 | 参数性质 | 功能说明                     |
| ---------------- | ------ | ------ | -------- | ---------------------------- |
| camera_pose      | STRING | Static | 必选     | 相机运动类型                 |
| speed            | FLOAT  | 1.0    | 必选     | 相机运动速度                 |
| video_length     | INT    | 16     | 必选     | 视频帧数                     |

**输出：**

* `CameraPose`: 相机运动轨迹。

**使用场景：**
- 创建简单的相机运动效果。

---

### CameraTrajectoryFromChaoJie

**描述：**
将 `CameraPose` 对象转换为相机轨迹参数的JSON字符串和视频长度，用于EasyAnimate的相机控制。

**输入参数表：**

| 参数名       | 类型       | 默认值      | 参数性质 | 功能说明         |
| ------------ | ---------- | ----------- | -------- | ---------------- |
| camera_pose  | CameraPose | -           | 必选     | 相机运动姿态对象 |
| fx           | FLOAT      | 0.474812461 | 可选     | x轴焦距        |
| fy           | FLOAT      | 0.844111024 | 可选     | y轴焦距        |
| cx           | FLOAT      | 0.5         | 可选     | x轴光学中心    |
| cy           | FLOAT      | 0.5         | 可选     | y轴光学中心    |

**输出：**

* `camera_trajectory` (STRING): 相机轨迹参数的JSON字符串。
* `video_length` (INT): 视频长度（帧数）。

**使用场景：**
- 将通过 `CameraBasicFromChaoJie`、`CameraCombineFromChaoJie` 或 `CameraJoinFromChaoJie` 生成的相机姿态转换为EasyAnimate V5+ V2V采样器可用的 `camera_conditions` 输入。

---

### CameraCombineFromChaoJie

**描述：**
合并多个相机运动轨迹。

**输入参数表：**

| 参数名           | 类型   | 默认值 | 参数性质 | 功能说明                     |
| ---------------- | ------ | ------ | -------- | ---------------------------- |
| camera_pose1     | STRING | Static | 必选     | 第一个相机运动轨迹           |
| camera_pose2     | STRING | Static | 必选     | 第二个相机运动轨迹           |
| camera_pose3     | STRING | Static | 可选     | 第三个相机运动轨迹           |
| camera_pose4     | STRING | Static | 可选     | 第四个相机运动轨迹           |
| speed            | FLOAT  | 1.0    | 必选     | 相机运动速度                 |
| video_length     | INT    | 16     | 必选     | 视频帧数                     |

**输出：**

* `CameraPose`: 合并后的相机运动轨迹。

**使用场景：**
- 创建复杂的相机运动效果。

---

### CameraJoinFromChaoJie

**描述：**
连接两个相机运动姿态 (`CameraPose`) 对象，生成一个新的组合相机运动姿态。

**输入参数表：**

| 参数名       | 类型       | 默认值 | 参数性质 | 功能说明           |
| ------------ | ---------- | ------ | -------- | ------------------ |
| camera_pose1 | CameraPose | -      | 必选     | 第一个相机运动姿态 |
| camera_pose2 | CameraPose | -      | 必选     | 第二个相机运动姿态 |

**输出：**

* `CameraPose`: 连接后的相机运动姿态对象。

**使用场景：**
- 顺序连接多个简单的相机运动，形成更复杂的相机运动序列。
- 例如，先平移再旋转。

---

### ImageMaximumNode

**描述：**
合并两段视频帧，取每帧的最大值。

**输入参数表：**

| 参数名           | 类型   | 默认值 | 参数性质 | 功能说明                     |
| ---------------- | ------ | ------ | -------- | ---------------------------- |
| video_1          | IMAGE  | -      | 必选     | 第一段视频帧                 |
| video_2          | IMAGE  | -      | 必选     | 第二段视频帧                 |

**输出：**

* `IMAGE`: 合并后的视频帧。

**使用场景：**
- 合并两段视频帧以生成特殊效果。

---

## 🔧 常用工作流组合

| 工作流名称         | 节点组合                                                                 | 用途                     |
| ------------------ | ----------------------------------------------------------------------- | ------------------------ |
| 文本到视频生成     | Load EasyAnimate Model → EasyAnimate_TextBox → EasyAnimate Sampler for T2V | 根据文本生成视频         |
| V5+文本到视频      | Load EasyAnimate Model → EasyAnimate_TextBox → EasyAnimateV5 Sampler for T2V | V5+模型文本生成视频      |
| 图像到视频生成     | Load EasyAnimate Model → EasyAnimate_TextBox → EasyAnimate Sampler for I2V | 从图像生成视频           |
| V5+图像到视频      | Load EasyAnimate Model → EasyAnimate_TextBox → EasyAnimateV5 Sampler for I2V | V5+模型图像生成视频      |
| 视频风格迁移       | Load EasyAnimate Model → EasyAnimate_TextBox → EasyAnimate Sampler for V2V | 对视频进行风格迁移       |
| V5+视频风格迁移(相机控制) | Load EasyAnimate Model → EasyAnimate_TextBox → CameraBasic/Combine/Join → CameraTrajectory → EasyAnimateV5 Sampler for V2V | V5+模型视频迁移并控制相机 |
| V5+视频风格迁移(轨迹控制) | Load EasyAnimate Model → EasyAnimate_TextBox → [KJNodes for Coords/Masks] → Create Trajectory → EasyAnimateV5 Sampler for V2V | V5+模型视频迁移并控制轨迹 |
| 复杂相机运动       | CameraBasicFromChaoJie → CameraJoinFromChaoJie → CameraCombineFromChaoJie → CameraTrajectoryFromChaoJie | 创建复杂相机运动参数     |
