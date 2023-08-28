# Generating Chest X-ray images via Large Language Models


## Getting Started
### Installation

**1. Prepare the code and the environment**

Git clone our repository and install the requirements.

```bash
git clone https://github.com/wang-zhanyu/SwinLLama.git
cd xray_gen
pip install -r requirements.txt
```

**2. Prepare the training dataset**

put mimic_cxr dataset under data folder. The data structure like this:

- data/
  - mimic_cxr
    - images
    - annotation.json
    - feature.pickle



### Training

To launch the training, run the following command.

```bash
bash start.sh
```