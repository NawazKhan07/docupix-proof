# How to Run DocPix (Artwork Compare)

A complete step-by-step guide to installing, configuring, and running the **DocPix Artwork Comparison Application** on your computer.

---

## 1. Prerequisites

Before running the application, make sure you have:

### A. Python 3.11+
* Ensure Python 3.11, 3.12, or 3.13 is installed.
* Check your version in PowerShell:
  ```powershell
  python --version
  ```

### B. Tesseract OCR (Required for text/number comparisons)
The app uses local Tesseract OCR to read and compare text without network calls.
* **Windows Installer:** Download and install the latest 64-bit installer from [UB-Mannheim Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki).
* Install to the standard directory: `C:\Program Files\Tesseract-OCR`.
* The app automatically detects Tesseract in this location, on your `PATH`, or from the `TESSERACT_CMD` environment variable.

---

## 2. Installation & Setup

### Step 1: Open Terminal in the Project Directory
Open **PowerShell** or **Command Prompt** and navigate to the project directory:
```powershell
cd d:\docpix
```

### Step 2: Create a Virtual Environment (Recommended)
Creating an isolated environment keeps dependencies clean:
```powershell
python -m venv .venv
```

### Step 3: Activate the Virtual Environment
* **On Windows (PowerShell):**
  ```powershell
  .venv\Scripts\Activate.ps1
  ```
  *(If you get a script execution policy error, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` and try again).*
* **On Windows (Command Prompt):**
  ```cmd
  .venv\Scripts\activate.bat
  ```
* **On macOS / Linux:**
  ```bash
  source .venv/bin/activate
  ```

### Step 4: Install Dependencies
Install the package in editable mode with development tools:
```powershell
pip install -e ".[dev]"
```

---

## 3. Launching the Application

With your virtual environment activated, run:

```powershell
python -m app.main
```
*(Or via the registered CLI command: `artwork-compare`)*

The main graphical user interface (PySide6 / Qt) will open on your screen.

---

## 4. How to Use the App: Step-by-Step

### Step 1: Load Files
1. In the **Original** box (approved artwork), click **Browse** or drag and drop your reference file (PDF, PNG, JPG, TIFF, BMP).
2. In the **Suspect** box (artwork under review), click **Browse** or drag and drop the comparison file.

### Step 2: (Optional) Isolate Artwork Area (Crop / Lift)
* If your sheet contains extra margins, slugs, colour control strips, or crop marks:
  * Click **Edit Artwork Area** on the Original slot.
  * Use **Crop** to draw a clean rectangular box around the artwork, or **Lift Subject** to automatically detect the die-cut border.
  * Click **Apply & Save**.

### Step 3: Select Comparison Mode
Choose the appropriate mode for your files:
* **Automatic:** Best for two digital files. Compares text, colours, and graphics with panel-wise registration.
* **Scan vs Digital:** Compares a physical printed scan against a digital source. Disables colour sensitivity and focuses on text/barcode marks.
* **Barcode / Pharmacode / QR:** Decodes and compares all 1D/2D barcodes, pharmacodes, and QR codes.
* **Braille:** Decodes and verifies embossed or printed Braille dots cell by cell.
* **Series:** Compares a single original artwork against multiple repeated copies on a press sheet.
* **Leaflet:** Compares multi-page documents word-by-word across pages.

### Step 4: Compare
Click the primary **Compare** button. The comparison will execute asynchronously in a background worker process while keeping the UI responsive.

### Step 5: Review Differences
Once complete, the **Results View** opens:
* **View Modes:** Toggle between **Side by Side**, **Flicker**, **Overlay**, and **Grayscale** in the toolbar.
* **Zoom & Pan:** Use the mouse wheel to zoom in (vector quality rendering is maintained at all zoom levels).
* **Findings List:** Step through detected differences on the right panel using the **Up / Down** arrow keys or by clicking on findings.

---

## 5. Running Tests & Self-Verification

### Verify Installation
To run the built-in offline self-test:
```powershell
python -m app.self_test
```

### Run the Automated Test Suite
To run the automated tests via pytest:
```powershell
python -m pytest tests/test_ocr_locator.py tests/test_detectors.py
```
To run the complete test suite:
```powershell
python -m pytest
```

---

## 6. Troubleshooting FAQs

* **"Tesseract OCR was not found"**
  Ensure Tesseract is installed at `C:\Program Files\Tesseract-OCR\tesseract.exe`. If installed elsewhere, set the environment variable:
  ```powershell
  $env:TESSERACT_CMD = "C:\Path\To\tesseract.exe"
  ```
* **"Cannot activate virtual environment (Execution of scripts is disabled)"**
  Run:
  ```powershell
  Set-ExecutionPolicy -Scope Process RemoteSigned
  ```
* **PySide6 display scaling issues**
  DocPix handles high-DPI scaling automatically. If you experience display scaling issues on multi-monitor setups, run:
  ```powershell
  $env:QT_ENABLE_HIGHDPI_SCALING = "1"
  python -m app.main
  ```
