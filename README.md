# A Simple Code Base for Gaussian Splatting Research

A lightweight and flexible codebase for Gaussian Splatting research,
designed for easy setup and configuration, making it ideal for quick
experimentation and prototyping.

## Installation

- **Option 1: Install Directly as a Library**

  ```bash
  pip install git+https://github.com/Awesome3DGS/libgs.git
  ```

- **Option 2: Clone and Install Locally**

  ```bash
  git clone https://github.com/Awesome3DGS/libgs.git && cd libgs
  pip install .      # Core dependencies
  pip install .[dev] # With dev tools for code formatting and linting (optional)
  ```

## Usage

The following usage instructions are based on the `base` pipeline, which is a foundational setup for replicating the standard 3D Gaussian Splatting. In addition to `base`, the pipeline directory includes several other pipelines that replicate different approaches.

#### 1. Setting Up Configuration

Use the following command to generate a configuration file with default parameters:

```
python main.py --pipeline=base --print > config/base.yaml
```

You can modify `base.yaml` to suit specific experimental setups.

#### 2. Running an Experiment

Start an experiment using the configuration file, with optional overrides from the command line (e.g., data.root):

```
python main.py --pipeline=base --config=config/base.yaml data.root=/mydata/myscene
```

*Configuration Priority*: Defaults in code < Config file < Command line

To explore other pipelines, replace base with the desired pipeline name. These pipelines offer useful examples and can guide the implementation of custom pipelines.

## Dataset

Currently, LibGS supports datasets in COLMAP and Blender formats. Below are the expected directory structures for different types of scenes:

#### 1. Static Blender Dataset

  ```
  .
  ├── test
  │   └── r_0.png
  ├── train
  │   └── r_0.png
  ├── transforms_test.json
  ├── transforms_train.json
  ├── transforms_val.json
  └── val
      └── r_0.png
  ```

#### 2. Static COLMAP Dataset

  ```
  .
  ├── images
  │   ├── cam00.png
  │   └── cam01.png
  └── sparse
      └── 0
          ├── cameras.bin
          ├── images.bin
          └── points3D.bin
  ```

#### 3. Dynamic COLMAP Dataset with Fixed Cameras

  ```
  .
  ├── frames
  │   ├── 0000
  │   │   ├── cam00.png
  │   │   └── cam01.png
  │   └── 0001
  │       ├── cam00.png
  │       └── cam01.png
  └── sparse
      └── 0
          ├── cameras.bin
          ├── images.bin
          └── points3D.bin
  ```

For datasets in other formats, please convert them to one of the above-supported structures for compatibility with LibGS.

## Acknowledgments

This codebase is primarily intended for internal use within our research group and still has many areas that need improvement. It draws heavily on prior work, and we gratefully acknowledge the following projects for their contributions:

- **gaussian-splatting**: https://github.com/graphdeco-inria/gaussian-splatting
- **4DGaussians**: https://github.com/hustvl/4DGaussians
- **Scaffold-GS**: https://github.com/city-super/Scaffold-GS

Please be mindful of these projects’ licenses if you utilize this code. If you encounter any issues or have suggestions for improvement, feel free to open an issue to provide feedback.
