# Config Directory README

## Overview

This directory contains configuration scripts essential for managing and organizing the dataset for various ethical frameworks. Below is a brief description of each script:

1. **ethical_frameworks.py:**  Integrates the functionality of `paths.py` and `prompt.py` to combine and manage the ethical datasets.

2. **paths.py:** Defines the directory structure and paths for the dataset files.

3. **prompt.py:** Contains all the prompts used for generating the dataset.

## Setting Up `paths.py`

The most critical aspect of `paths.py` is the `EASYEDIT_PATH` variable. This variable should point to the directory where the [EasyEdit repository](https://github.com/zjunlp/EasyEdit) is cloned.

### Example Configuration

The default value is set to `/app`. If your EasyEdit repository is cloned elsewhere, update this variable accordingly.

```
EASYEDIT_PATH = '/path/to/your/easyedit/repo'
```
