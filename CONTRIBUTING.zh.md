# 贡献指南

欢迎您为 Awesome-ComfyUI-Video 项目做出贡献！我们非常感谢您的帮助。

> 本指南的英文版本，请参阅 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 目录

- [行为准则](#行为准则)
- [如何开始](#如何开始)
- [文件夹结构](#文件夹结构)
- [如何贡献](#如何贡献)
  - [报告问题](#报告问题)
  - [提交功能请求](#提交功能请求)
  - [添加新的自定义节点文档](#添加新的自定义节点文档)
  - [翻译现有文档](#翻译现有文档)
  - [更新 README 和节点列表](#更新-readme-和节点列表)
- [Pull Request (PR) 指南](#pull-request-pr-指南)

## 行为准则

我们致力于提供一个友好和包容的环境。所有贡献者都应遵守项目的[行为准则](CODE_OF_CONDUCT.md)（如果项目有的话，否则可以暂时移除此链接或后续添加）。

## 如何开始

1.  **Fork 本仓库**：点击页面右上角的 "Fork" 按钮。
2.  **Clone您的 Fork**：`git clone https://github.com/YOUR_USERNAME/Awesome-ComfyUI-Video.git`
3.  **创建分支**：`git checkout -b your-feature-branch`
4.  进行修改并提交。
5.  **Push到您的Fork**：`git push origin your-feature-branch`
6.  创建 Pull Request 到主仓库的 `main` 分支。

## 文件夹结构

为了支持多语言文档，我们采用了以下结构：

```
Awesome-ComfyUI-Video/
├── docs/
│   ├── en/  # 英文文档 (主要)
│   │   ├── custom_nodes/  # 英文自定义节点详情
│   │   │   └── some_node_en.md
│   │   └── templates/
│   │       └── custom_nodes_template_en.md # 英文节点文档模板
│   ├── zh/  # 中文文档
│   │   ├── custom_nodes/  # 中文自定义节点详情
│   │   │   └── videohelpersuite_zh.md (示例)
│   │   └── templates/
│   │       └── custom_nodes_template_zh.md # 中文节点文档模板
│   └── nodes.md  # (未来可能也需要多语言版本或重构)
├── README.md
├── CONTRIBUTING.md  # 英文版贡献指南
└── CONTRIBUTING.zh.md # 本文件 (中文版贡献指南)
```

**语言优先级：**
- **英文 (en):** 主要和默认语言。所有新的节点文档应首先提供英文版本。
- **中文 (zh):** 欢迎提供中文翻译和中文原创内容。

我们鼓励贡献者尽可能提供双语（英文和中文）文档。

## 如何贡献

### 报告问题

如果您发现了 bug，请通过 GitHub Issues 提交。请提供详细信息，包括：
-   复现步骤。
-   期望行为和实际行为。
-   相关截图或日志。

### 提交功能请求

如果您有新功能或改进建议，也请通过 GitHub Issues 提交。

### 添加新的自定义节点文档

1.  **准备内容**：
    *   **英文版 (必需)**: 在 `docs/en/custom_nodes/` 目录下，使用 `docs/en/templates/custom_nodes_template_en.md` 模板创建文档。文件名应为节点包的小写名称，用连字符分隔 (例如 `your-node-set-name.md`)。
    *   **中文版 (强烈推荐)**: 在 `docs/zh/custom_nodes/` 目录下，使用 `docs/zh/templates/custom_nodes_template_zh.md` 模板创建对应的中文文档。文件名与英文版保持一致。
2.  **填写内容**：按照模板的结构详细填写节点信息。确保信息准确、清晰。
3.  **更新列表**：
    *   在主 `README.md` 的 `## Outstanding Community Custom Nodes` 表格中添加一行。`Details Doc` 链接应优先指向英文文档，如果中文文档也已提供，可以在描述中或另起一行注明。

### 翻译现有文档

如果您希望将现有的英文文档翻译成中文，或者将中文文档翻译成英文（如果英文版缺失）：

1.  找到一篇需要翻译的文档。
2.  在目标语言的 `custom_nodes` 目录下创建同名文件。
    *   例如，翻译 `docs/en/custom_nodes/some_node.md` 到中文，则创建 `docs/zh/custom_nodes/some_node.md`。
3.  翻译文档内容，保持格式和信息的准确性。
4.  提交 Pull Request 时请注明是翻译工作。

### 更新 README 和节点列表

-   当添加新的主要自定义节点包文档时，请在主 `README.md` 的表格中添加条目。
    -   `Details Doc` 链接应指向英文文档路径 (例如 `[Details](docs/en/custom_nodes/your-node.md)`)。
    -   如果中文文档也可用，可以在同一单元格内提供链接，例如 `[EN Details](docs/en/custom_nodes/your-node.md) / [中文详情](docs/zh/custom_nodes/your-node.md)`，或者在节点描述中提及。

## Pull Request (PR) 指南

-   确保您的 PR 标题清晰描述了更改内容。
-   在 PR 描述中提供更改的简要说明和动机。
-   如果您的 PR 解决了某个 Issue，请在描述中链接该 Issue (例如 `Closes #123`)。
-   确保您的代码或文档遵循项目现有的风格。
-   在提交 PR 前，请在本地测试您的更改。

感谢您的贡献！
