
# Easy-PyTorch


<p align="center">
  中文 | <a href="README_en.md">English</a>
</p>


![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fraw.githubusercontent.com%2FWangQvQ%2FEasy-PyTorch%2Fmain%2Fpackage.json&query=version&prefix=v&label=Easy-PyTorch) ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fraw.githubusercontent.com%2FWangQvQ%2FEasy-PyTorch%2Fmain%2Fpackage.json&query=license&label=license&color=%237FFF00) ![Visual Studio Marketplace Downloads](https://img.shields.io/visual-studio-marketplace/d/WangQvQ.Easy-PyTorch?color=red)


**Easy-PyTorch** 是一个为 VSCode 用户设计的插件，基于早先的 [vscode-pytorch](https://github.com/SvenBecker/vscode-pytorch) 项目。我们重启这个项目，让它与 PyTorch 最新版本保持同步，从而提供更好的支持。这个插件是编程新手的好帮手，让使用 PyTorch 的过程变得简单直观。

## 项目简介

通过这个插件，用户可以快速插入预设的代码片段，这些片段涵盖了从基础的张量操作到复杂的神经网络结构的各种常用代码模板。只需要简单的命令或几个点击，即可将这些模板代码插入到你的工程中，极大地提高编写效率。

此外，Easy-PyTorch 还提供了一系列教程和提示，这些内容适合所有水平的开发者，特别是初学者。这些教程会引导你理解每个代码片段的作用和应用场景，帮助你更好地掌握 PyTorch 的使用。

我们的目标是让机器学习的编程不仅高效，而且更加用户友好，即使是编程新手也能快速上手并享受创造自己的机器学习模型的乐趣。

<div align="center">
<img src="https://raw.githubusercontent.com/WangQvQ/Easy-PyTorch/main/images/preview.gif" alt="peculiarity"/>
</div>


## 特性

- **代码片段库**：涵盖各种 PyTorch 操作的丰富代码片段，提供快速编程的捷径。
- **智能提示功能**：在您编写代码时，只需输入 `pytorch`，就能看到与之相关的所有可用代码片段。
- **高度定制**：插件的代码片段包括占位符、变量和选项，您可以轻松定制，并通过 Tab 键在不同选项间切换。


<div align="center">
<img src="https://raw.githubusercontent.com/WangQvQ/Easy-PyTorch/main/images/prompt.png" alt="peculiarity" width="800"/>
</div>

<div align="center">
<img src="https://raw.githubusercontent.com/WangQvQ/Easy-PyTorch/refs/heads/main/images/catalogue.png" alt="peculiarity"/>
</div>

## 如何安装



### 方法 1
Visual Studio Code 扩展中搜索 Easy Pytorch 点击安装即可

<div align="center">
<img src="https://raw.githubusercontent.com/WangQvQ/Easy-PyTorch/main/images/vscode.png" alt="peculiarity"/>
</div>



### 方法 2
将本地项目移动到 Visual Studio Code 的扩展文件夹，可以通过简单的文件操作完成。以下是一般步骤和注意事项：

1. 确定 VSCode 扩展文件夹位置
Visual Studio Code 扩展通常存放在以下位置：
- **Windows**: `C:\Users\{你的用户名}\.vscode\extensions`
- **macOS**: `~/.vscode/extensions`
- **Linux**: `~/.vscode/extensions`

2. 下载并移动项目

- **手动下载**: 手动下载并解压到步骤 1 的路径下。



## 如何使用

- 在 Python 文件中输入 `pytorch` 开始，系统会显示所有相关的代码片段。
- 或直接键入 `conv` `pooling` 等代码，支持模糊输入。
- 从弹出的列表中选择所需片段，按 Enter 键插入。
- 按 Tab 键在代码片段的各个占位符之间自由切换。

这个插件旨在让您的机器学习项目起步更快，同时也让编程学习过程更加轻松。无论您是 PyTorch 新手还是希望提高编程效率的开发者，Easy-PyTorch 都是您的理想选择。