# AI Image Attack & Defense Dashboard

A Streamlit-based application for testing adversarial image attacks and evaluating defense techniques using AI vision models.

## Features

- Upload PNG or JPEG images
- Apply adversarial attacks:
  - Gaussian Noise
  - Pixel Shift
  - EXIF Prompt Injection
  - LSB Prompt Injection
- Apply defense techniques:
  - EXIF Metadata Removal
  - Gaussian Blur
- Test images using:
  - ViT Image Classifier
  - LLaVA Vision Model
- Compare:
  - Original Image
  - Attacked Image
  - Defended Image
- View confidence scores and attack evaluation
- Download the processed image

---

## Prerequisites

- Python 3.10 or later
- Git
- VS Code (recommended)

For LLaVA support:

- Install Ollama
- Pull the LLaVA model

```bash
ollama pull llava
```

---

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd <repository-folder>
```

### 2. Create a virtual environment

Windows

```bash
python -m venv venv
venv\Scripts\activate
```

Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Run the Application

```bash
streamlit run app.py
```
## Project Structure

```
project/
│
├── app.py
├── requirements.txt
├── README.md
├── modules/
│   ├── stego_attack.py
│   ├── vision_defense.py
│   ├── vision_target.py
│   └── vision_target_llava.py
│──Benchmark/
|   └── Test_report_hiba.xlx
|
│──Results/
|    ├── Attack success rate.png
|    ├── defense recovery rate.png
|    └── Attack vs Defense success rate.png
|    └── Test_report_hiba.xlx
|
│──Screen shots/
│──Sample images/
│──Adversarial_vision_lab_pitch.pptx

   
```

## Usage

1. Upload a PNG or JPEG image.
2. Select an attack method.
3. Adjust attack parameters.
4. (Optional) Enable defense filters.
5. Select the target model (ViT or LLaVA).
6. View predictions, confidence scores, and evaluation results.
7. Download the processed image if needed.

---

## Notes

- Maximum supported upload size: **20 MB**
- Images larger than **1024 pixels** are automatically resized for faster processing.
- LLaVA requires a locally running Ollama instance.
- ViT works without Ollama.

---

## Technologies Used

- Python
- Streamlit
- Pillow
- NumPy
- Pandas
- Transformers (ViT)
- Ollama (LLaVA)

---

## Authors

Internship Project

AI Image Attack & Defense Dashboard

Intern - Hiba Fathima K
