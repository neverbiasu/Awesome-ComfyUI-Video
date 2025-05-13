# 📹 ComfyUI-VideoHelperSuite

## 🔍 集合概览

| 项目       | 详情                                                                  |
| ---------- | --------------------------------------------------------------------- |
| 📌 **作者** | [@Kosinkadink](https://github.com/Kosinkadink)                        |
| 📅 **版本** | 1.2.0+                                                                |
| 🏷️ **分类** | 视频处理                                                              |
| 🔗 **仓库** | [GitHub链接](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) |

## 📝 功能简介
视频工作流辅助工具集，提供视频导入导出、帧处理与预览功能，是视频生成和编辑的核心辅助工具。

## 📊 节点一览表

| 节点名称                 | 类型 | 主要功能                 | 复杂度 | 常用场景               |
| ------------------------ | ---- | ------------------------ | ------ | ---------------------- |
| **Load Video (Upload)**  | I/O  | 上传并转换视频为图像序列 | ★★☆☆☆  | 本地视频导入处理       |
| **Load Video (Path)**    | I/O  | 从路径加载视频为图像序列 | ★★☆☆☆  | 批量视频处理           |
| **Load Video FFmpeg**    | I/O  | 使用FFmpeg加载视频       | ★★☆☆☆  | 特殊格式视频导入       |
| **Video Combine**        | I/O  | 将图像序列合成为视频     | ★★☆☆☆  | 视频导出、创建循环     |
| **Load Images**          | I/O  | 加载图像序列             | ★☆☆☆☆  | 处理已有图像帧         |
| **Load Audio**           | I/O  | 加载音频文件             | ★☆☆☆☆  | 为视频添加音轨         |
| **Split Images/Latents** | 处理 | 将批次分为两部分         | ★☆☆☆☆  | 长视频分段处理         |
| **Merge Images/Latents** | 处理 | 合并多组图像/潜空间      | ★☆☆☆☆  | 视频片段合并           |
| **Select Every Nth**     | 处理 | 按间隔选择帧             | ★☆☆☆☆  | 减少帧数、创建延时效果 |
| **Video Info**           | 工具 | 显示视频详细信息         | ★☆☆☆☆  | 分析视频参数           |
| **Meta Batch Manager**   | 高级 | 管理批处理操作           | ★★★☆☆  | 处理超长视频           |

## 📑 核心节点详解

### Load Video (Upload)

| 参数名             | 类型  | 默认值     | 参数性质 | 功能说明                |
| ----------------- | ----- | ---------- | -------- | ----------------------- |
| video             | FILE  | -          | 必选     | 要加载的视频文件        |
| force_rate        | FLOAT | 0.0        | 可选     | 强制调整帧率，设为0禁用 |
| force_size        | COMBO | "Disabled" | 可选     | 快速调整尺寸选项        |
| frame_load_cap    | INT   | 0          | 可选     | 最大返回帧数(批次大小)  |
| skip_first_frames | INT   | 0          | 可选     | 跳过开头的帧数          |
| select_every_nth  | INT   | 1          | 可选     | 每N帧采样一帧           |

**输出**: IMAGE[] (图像序列), VHS_VIDEOINFO (视频信息), AUDIO (可选音频)

**使用场景**:
- 将视频分解为帧进行AI处理
- 准备视频用于AnimateDiff处理
- 对视频进行分段处理

### Load Video (Path)

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

### Video Combine

| 参数名           | 类型    | 默认值        | 参数性质 | 功能说明                 |
| --------------- | ------- | ------------- | -------- | ------------------------ |
| images          | IMAGE[] | -             | 必选     | 要合成的图像序列         |
| frame_rate      | FLOAT   | 8.0           | 必选     | 输出视频帧率             |
| loop_count      | INT     | 0             | 可选     | 视频重复次数(0=无限循环) |
| filename_prefix | STRING  | "AnimateDiff" | 可选     | 输出文件名前缀           |
| format          | COMBO   | "h264-mp4"    | 必选     | 视频格式选择             |
| pingpong        | BOOLEAN | False         | 可选     | 创建往返循环效果         |
| save_output     | BOOLEAN | True          | 可选     | 是否保存到输出目录       |
| audio           | AUDIO   | -             | 条件可选 | 可选音频轨道             |

**输出**: VHS_FILENAMES (生成文件路径列表)

**格式选项**:
- h264-mp4: 通用兼容格式
- vp9-webm: 网页友好格式
- gif: 动画GIF格式
- webp: 高压缩动画格式
- png-sequence: 无损图像序列

**使用场景**:
- 将AI处理后的帧合成为视频
- 创建循环动画效果
- 为视频添加音频轨道

### Load Audio (Path)

| 参数名        | 类型   | 默认值   | 参数性质 | 功能说明               |
| ------------ | ------ | -------- | -------- | ---------------------- |
| audio_file   | STRING | "input/" | 必选     | 音频文件路径           |
| seek_seconds | FLOAT  | 0        | 可选     | 从指定时间开始加载(秒) |

**输出**: AUDIO (音频数据)

**支持格式**: mp3, wav, ogg, m4a, flac

**使用场景**:
- 为视频添加配乐
- 从视频中提取的音频重新应用
- 创建带音效的动画

### Split Images

| 参数名       | 类型    | 默认值 | 参数性质 | 功能说明             |
| ----------- | ------- | ------ | -------- | -------------------- |
| images      | IMAGE[] | -      | 必选     | 要分割的图像序列     |
| split_index | INT     | 1      | 必选     | 分割点(前N个到输出A) |

**输出**: IMAGE[] (前半部分), IMAGE[] (后半部分)

**使用场景**:
- 将长视频分为多段处理
- 对视频前后部分应用不同效果
- 分离视频的特定部分进行处理

### Video Info

| 参数名      | 类型          | 默认值 | 参数性质 | 功能说明     |
| ---------- | ------------- | ------ | -------- | ------------ |
| video_info | VHS_VIDEOINFO | -      | 必选     | 视频信息对象 |

**输出**: 
- 源视频信息(fps, frame_count, duration, width, height)
- 加载后视频信息(fps, frame_count, duration, width, height)

**使用场景**:
- 分析视频参数
- 根据视频特性调整处理参数
- 显示处理前后的变化

### Meta Batch Manager

| 参数名            | 类型 | 默认值 | 参数性质 | 功能说明           |
| ---------------- | ---- | ------ | -------- | ------------------ |
| frames_per_batch | INT  | 16     | 必选     | 每批处理的最大帧数 |

**输出**: VHS_BatchManager (批处理管理器)

**使用场景**:
- 处理超长视频
- 控制显存使用量
- 高级批处理工作流

## 🔧 常用工作流组合

| 工作流名称     | 节点组合                                                            | 用途                   |
| -------------- | ------------------------------------------------------------------- | ---------------------- |
| 长视频分段处理 | Load Video → Meta Batch Manager → 处理 → Video Combine              | 处理超长视频           |
| 视频加音频     | Load Video + Load Audio → 处理 → Video Combine                      | 保留或添加音轨         |
| 循环动画创建   | Load Video → 处理 → Video Combine(pingpong=True)                    | 创建来回循环效果       |
| 选择性处理     | Load Video → Split Images → 分别处理 → Merge Images → Video Combine | 对不同部分应用不同效果 |
