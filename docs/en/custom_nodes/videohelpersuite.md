# 📹 ComfyUI-VideoHelperSuite

## 🔍 Overview

| Item        | Details                                                                  |
| ----------- | ------------------------------------------------------------------------ |
| 📌 **Author** | [@Kosinkadink](https://github.com/Kosinkadink)                           |
| 📅 **Version** | 1.2.0+                                                                   |
| 🏷️ **Category** | Video Processing                                                         |
| 🔗 **Repository** | [GitHub Link](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) |

## 📝 Introduction
A collection of auxiliary tools for video workflows, providing video import/export, frame processing, and preview functions. It is a core auxiliary tool for video generation and editing.

## 📊 Node List

| Node Name                | Type    | Main Function                                  | Complexity | Common Use Cases             |
| ------------------------ | ------- | ---------------------------------------------- | ---------- | ---------------------------- |
| **Load Video (Upload)**  | I/O     | Upload and convert video to image sequence     | ★★☆☆☆      | Local video import processing|
| **Load Video (Path)**    | I/O     | Load video from path as image sequence         | ★★☆☆☆      | Batch video processing       |
| **Load Video FFmpeg**    | I/O     | Load video using FFmpeg                        | ★★☆☆☆      | Special format video import  |
| **Video Combine**        | I/O     | Combine image sequence into video              | ★★☆☆☆      | Video export, create loops   |
| **Load Images**          | I/O     | Load image sequence                            | ★☆☆☆☆      | Process existing image frames|
| **Load Audio**           | I/O     | Load audio file                                | ★☆☆☆☆      | Add audio track to video     |
| **Split Images/Latents** | Process | Split batch into two parts                     | ★☆☆☆☆      | Segment long videos          |
| **Merge Images/Latents** | Process | Merge multiple groups of images/latents        | ★☆☆☆☆      | Video segment merging        |
| **Select Every Nth**     | Process | Select frames at intervals                     | ★☆☆☆☆      | Reduce frame rate, time-lapse|
| **Video Info**           | Utility | Display detailed video information             | ★☆☆☆☆      | Analyze video parameters     |
| **Meta Batch Manager**   | Advanced| Manage batch processing operations             | ★★★☆☆      | Process very long videos     |

## 📑 Core Node Details

### Load Video (Upload)

| Parameter Name      | Type  | Default Value | Property | Description                             |
| ----------------- | ----- | ------------- | -------- | --------------------------------------- |
| video             | FILE  | -             | Required | Video file to load                      |
| force_rate        | FLOAT | 0.0           | Optional | Force frame rate (0 to disable)         |
| force_size        | COMBO | "Disabled"    | Optional | Quick resize options                    |
| frame_load_cap    | INT   | 0             | Optional | Max frames to return (batch size)       |
| skip_first_frames | INT   | 0             | Optional | Skip N frames from the beginning        |
| select_every_nth  | INT   | 1             | Optional | Sample one frame every N frames         |

**Output**: IMAGE[] (Image sequence), VHS_VIDEOINFO (Video information), AUDIO (Optional audio)

**Use Cases**:
- Decompose video into frames for AI processing
- Prepare video for AnimateDiff processing
- Process video in segments

### Load Video (Path)

| Parameter Name      | Type   | Default Value | Property | Description                             |
| ----------------- | ------ | ------------- | -------- | --------------------------------------- |
| video_path        | STRING | "input/"      | Required | Video file path                         |
| force_rate        | FLOAT  | 0.0           | Optional | Force frame rate (0 to disable)         |
| force_size        | COMBO  | "Disabled"    | Optional | Quick resize options                    |
| frame_load_cap    | INT    | 0             | Optional | Max frames to return (batch size)       |
| skip_first_frames | INT    | 0             | Optional | Skip N frames from the beginning        |
| select_every_nth  | INT    | 1             | Optional | Sample one frame every N frames         |

**Output**: IMAGE[] (Image sequence), VHS_VIDEOINFO (Video information), AUDIO (Optional audio)

**Use Cases**:
- Batch process videos from server paths
- Process videos from network URLs
- Use in automated workflows

### Video Combine

| Parameter Name    | Type    | Default Value | Property          | Description                      |
| --------------- | ------- | ------------- | ----------------- | -------------------------------- |
| images          | IMAGE[] | -             | Required          | Image sequence to combine        |
| frame_rate      | FLOAT   | 8.0           | Required          | Output video frame rate          |
| loop_count      | INT     | 0             | Optional          | Video loop count (0=infinite)    |
| filename_prefix | STRING  | "AnimateDiff" | Optional          | Output filename prefix           |
| format          | COMBO   | "h264-mp4"    | Required          | Video format selection           |
| pingpong        | BOOLEAN | False         | Optional          | Create a back-and-forth loop     |
| save_output     | BOOLEAN | True          | Optional          | Save to output directory         |
| audio           | AUDIO   | -             | Conditional Opt.  | Optional audio track             |

**Output**: VHS_FILENAMES (List of generated file paths)

**Format Options**:
- h264-mp4: Common compatible format
- vp9-webm: Web-friendly format
- gif: Animated GIF format
- webp: High-compression animated format
- png-sequence: Lossless image sequence

**Use Cases**:
- Combine AI-processed frames into a video
- Create looping animation effects
- Add an audio track to a video

### Load Audio (Path)

| Parameter Name | Type   | Default Value | Property | Description                          |
| ------------ | ------ | ------------- | -------- | ------------------------------------ |
| audio_file   | STRING | "input/"      | Required | Audio file path                      |
| seek_seconds | FLOAT  | 0             | Optional | Start loading from specified time (sec)|

**Output**: AUDIO (Audio data)

**Supported Formats**: mp3, wav, ogg, m4a, flac

**Use Cases**:
- Add background music to video
- Reapply audio extracted from video
- Create animations with sound effects

### Split Images

| Parameter Name | Type    | Default Value | Property | Description                           |
| ----------- | ------- | ------------- | -------- | ------------------------------------- |
| images      | IMAGE[] | -             | Required | Image sequence to split               |
| split_index | INT     | 1             | Required | Split point (first N to output A)     |

**Output**: IMAGE[] (First part), IMAGE[] (Second part)

**Use Cases**:
- Split long videos into multiple segments for processing
- Apply different effects to the beginning and end of a video
- Isolate specific parts of a video for processing

### Video Info

| Parameter Name | Type          | Default Value | Property | Description          |
| ---------- | ------------- | ------------- | -------- | -------------------- |
| video_info | VHS_VIDEOINFO | -             | Required | Video information object|

**Output**:
- Source video info (fps, frame_count, duration, width, height)
- Loaded video info (fps, frame_count, duration, width, height)

**Use Cases**:
- Analyze video parameters
- Adjust processing parameters based on video characteristics
- Display changes before and after processing

### Meta Batch Manager

| Parameter Name     | Type | Default Value | Property | Description                      |
| ---------------- | ---- | ------------- | -------- | -------------------------------- |
| frames_per_batch | INT  | 16            | Required | Max frames per batch to process  |

**Output**: VHS_BatchManager (Batch manager)

**Use Cases**:
- Process very long videos
- Control VRAM usage
- Advanced batch processing workflows

## 🔧 Common Workflow Combinations

| Workflow Name             | Node Combination                                                            | Purpose                                  |
| ------------------------- | --------------------------------------------------------------------------- | ---------------------------------------- |
| Long Video Segmentation   | Load Video → Meta Batch Manager → Process → Video Combine                   | Process very long videos                 |
| Add Audio to Video        | Load Video + Load Audio → Process → Video Combine                           | Preserve or add an audio track           |
| Create Looping Animation  | Load Video → Process → Video Combine(pingpong=True)                         | Create back-and-forth looping effects    |
| Selective Processing      | Load Video → Split Images → Process Separately → Merge Images → Video Combine | Apply different effects to different parts|
