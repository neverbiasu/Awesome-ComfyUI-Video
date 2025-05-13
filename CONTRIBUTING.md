# Contribution Guidelines

Welcome to Awesome-ComfyUI-Video! We appreciate your help in making this project better.

> For the Chinese version of this guide, please see [CONTRIBUTING.zh.md](CONTRIBUTING.zh.md).

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Folder Structure](#folder-structure)
- [How to Contribute](#how-to-contribute)
  - [Reporting Issues](#reporting-issues)
  - [Submitting Feature Requests](#submitting-feature-requests)
  - [Adding New Custom Node Documentation](#adding-new-custom-node-documentation)
  - [Translating Existing Documentation](#translating-existing-documentation)
  - [Updating README and Node Lists](#updating-readme-and-node-lists)
- [Pull Request (PR) Guidelines](#pull-request-pr-guidelines)

## Code of Conduct

We are committed to providing a friendly and inclusive environment. All contributors are expected to adhere to the project's [Code of Conduct](CODE_OF_CONDUCT.md) (if one exists, otherwise remove this link or add one later).

## Getting Started

1.  **Fork this repository**: Click the "Fork" button in the upper right corner of the page.
2.  **Clone your fork**: `git clone https://github.com/YOUR_USERNAME/Awesome-ComfyUI-Video.git`
3.  **Create a branch**: `git checkout -b your-feature-branch`
4.  Make your changes and commit them.
5.  **Push to your fork**: `git push origin your-feature-branch`
6.  Create a Pull Request to the `main` branch of the main repository.

## Folder Structure

To support multilingual documentation, we use the following structure:

```
Awesome-ComfyUI-Video/
├── docs/
│   ├── en/  # English documentation (Primary)
│   │   ├── custom_nodes/  # English custom node details
│   │   │   └── some_node_en.md
│   │   └── templates/
│   │       └── custom_nodes_template_en.md # English node documentation template
│   ├── zh/  # Chinese documentation
│   │   ├── custom_nodes/  # Chinese custom node details
│   │   │   └── videohelpersuite_zh.md (example)
│   │   └── templates/
│   │       └── custom_nodes_template_zh.md # Chinese node documentation template
│   └── nodes.md  # (May also require multilingual versions or refactoring in the future)
├── README.md
├── CONTRIBUTING.md  # This file (English)
└── CONTRIBUTING.zh.md # Chinese version of contribution guidelines
```

**Language Priority:**
- **English (en):** Primary and default language. All new node documentation should first be provided in English.
- **Chinese (zh):** Chinese translations and original Chinese content are welcome.

We encourage contributors to provide bilingual (English and Chinese) documentation whenever possible.

## How to Contribute

### Reporting Issues

If you find a bug, please submit it via GitHub Issues. Provide detailed information, including:
-   Steps to reproduce.
-   Expected behavior and actual behavior.
-   Relevant screenshots or logs.

### Submitting Feature Requests

If you have new features or improvement suggestions, please also submit them via GitHub Issues.

### Adding New Custom Node Documentation

1.  **Prepare Content**:
    *   **English Version (Required)**: In the `docs/en/custom_nodes/` directory, create the documentation using the `docs/en/templates/custom_nodes_template_en.md` template. The filename should be the lowercase name of the node pack, with hyphens for spaces (e.g., `your-node-set-name.md`).
    *   **Chinese Version (Highly Recommended)**: In the `docs/zh/custom_nodes/` directory, create the corresponding Chinese documentation using the `docs/zh/templates/custom_nodes_template_zh.md` template. The filename should match the English version.
2.  **Fill in Content**: Detail the node information according to the template structure. Ensure information is accurate and clear.
3.  **Update Lists**:
    *   Add a row to the table in the main `README.md` under `## Outstanding Community Custom Nodes`. The `Details Doc` link should primarily point to the English documentation. If a Chinese version is also available, it can be mentioned in the description or as an additional link.

### Translating Existing Documentation

If you wish to translate existing English documentation into Chinese, or Chinese documentation into English (if the English version is missing):

1.  Find a document that needs translation.
2.  Create a file with the same name in the target language's `custom_nodes` directory.
    *   For example, to translate `docs/en/custom_nodes/some_node.md` to Chinese, create `docs/zh/custom_nodes/some_node.md`.
3.  Translate the content, maintaining consistent formatting and accuracy.
4.  Please indicate in your Pull Request that it is a translation effort.

### Updating README and Node Lists

-   When adding documentation for a new major custom node pack, add an entry to the table in the main `README.md`.
    -   The `Details Doc` link should point to the English documentation path (e.g., `[Details](docs/en/custom_nodes/your-node.md)`).
    -   If a Chinese version is also available, you can provide a link in the same cell, e.g., `[EN Details](docs/en/custom_nodes/your-node.md) / [中文详情](docs/zh/custom_nodes/your-node.md)`, or mention it in the node description.

## Pull Request (PR) Guidelines

-   Ensure your PR title clearly describes the changes.
-   Provide a brief description and motivation for the changes in the PR description.
-   If your PR resolves an issue, link to that issue in the description (e.g., `Closes #123`).
-   Ensure your code or documentation follows the project's existing style.
-   Test your changes locally before submitting a PR.

Thank you for your contributions!
