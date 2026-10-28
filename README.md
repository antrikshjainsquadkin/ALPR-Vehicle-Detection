# Vehicle Detection and License Plate Recognition

A Python-based computer vision application that detects vehicle license plates from video footage and recognizes their characters using Optical Character Recognition (OCR).

The application is designed for local execution, with structured JSON files used to store recognized license plate records.

## Features

- **License Plate Detection:** Detect license plates in video frames using a YOLO-based detection model.
- **Optical Character Recognition (OCR):** Extract characters from detected license plates.
- **Video Processing:** Process video files and inspect detection results.
- **Live Stream Support:** Process supported RTSP camera streams.
- **Recognition Records:** Store recognized license plate numbers with timestamps in local JSON files.
- **Duplicate Reduction:** Use a time-window-based mechanism to reduce repeated records for the same normalized license plate.
- **Confidence-Based Recognition:** Use available detection and OCR confidence information to help assess recognition results.
- **Local Execution:** Run the application locally without requiring a cloud inference service or external API at runtime.
- **Preview Mode:** View video processing and detection results when preview is enabled.
- **Configurable Models:** Keep model files locally and configure the detection and OCR components for the project environment.

## How It Works

The application follows this processing workflow:

1. **Input Source:** Select a video file or provide a supported RTSP stream.
2. **Frame Processing:** Read frames from the input source for analysis.
3. **License Plate Detection:** Run the detection model to locate license plates in each frame.
4. **Plate Extraction:** Extract the detected plate region for text recognition.
5. **Text Recognition:** Apply OCR to recognize the characters on the plate.
6. **Recognition Assessment:** Evaluate the available recognition confidence and handle uncertain results according to the application's implementation.
7. **Duplicate Handling:** Compare normalized plate text and timestamps to reduce repeated records within the configured time window.
8. **Record Storage:** Save the resulting plate records to local JSON files.
9. **Preview and Output:** Display detection results when preview mode is enabled and retain the generated records for later review.

## Technology Stack

- **Python** — Application development
- **Ultralytics YOLO** — License plate detection
- **OCR Engine** — License plate text recognition using the OCR engine configured in the project
- **OpenCV** — Video and image frame processing, where used by the application
- **JSON** — Local storage of recognition records
- **PyTorch** — Model inference, where required by the configured detection or OCR components

## Project Structure

```text
Vehicle-Detection/
├── vehicle_license_plate_recognition.py
├── main.py
├── models/
│   └── license_plate_detector.pt
├── data/
│   └── recognized_license_plate_records.json
├── requirements.txt
├── .gitignore
└── README.md
```

The structure above is illustrative. Actual model files, OCR assets, configuration files, and supporting modules may differ in your local project.

## Requirements

- Windows, Linux, or another supported Python environment
- Python version compatible with the installed project dependencies
- Git
- Local license plate detection model
- Required OCR model files and dependencies
- Video input file or accessible RTSP stream

A compatible CPU can be used for inference. GPU acceleration depends on the hardware and installed framework configuration.

## Installation

### 1. Clone the repository

```powershell
git clone https://github.com/antrikshjainsquadkin/Vehicle-Detection.git
cd Vehicle-Detection
```

### 2. Create a virtual environment

```powershell
py -m venv .venv
```

### 3. Activate the virtual environment

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

If the repository contains a `requirements.txt` file:

```powershell
python -m pip install -r requirements.txt
```

If the file does not exist, install the dependencies specified by the project instead of installing arbitrary packages.

### 5. Configure model files

Ensure that the required detection model and OCR model assets are available at the paths expected by the application.

For example:

```text
models/license_plate_detector.pt
```

Use the model paths configured in your actual source code.

## Usage

### Process a video file

Run the application with a supported local video file:

```powershell
python vehicle_license_plate_recognition.py --source "C:\path\to\video.mp4"
```

### Run with preview enabled

```powershell
python vehicle_license_plate_recognition.py --source "C:\path\to\video.mp4" --preview
```

Replace the example path with your actual video path.

### Process an RTSP stream

If the application accepts RTSP URLs through the `--source` argument:

```powershell
python vehicle_license_plate_recognition.py --source "rtsp://<camera-address>/<stream-path>"
```

Replace the placeholder with your camera's RTSP address. Do not publish camera credentials or private stream URLs.

**Note:** Available command-line options depend on the arguments implemented in the current application. Check the source code or run the application's supported help command before using additional options.

## Output and Data Storage

Recognition records are stored in local JSON files rather than a database service.

A record may contain information such as:

- Recognized license plate text
- Detection or recognition timestamp
- Confidence information, when recorded by the implementation
- Recognition status or additional observations, when supported

Example of a simplified JSON record:

```json
{
  "plate_number": "GJ01AB1234",
  "timestamp": "2026-10-28T10:30:00",
  "confidence": 0.92
}
```

This example illustrates the format only. Actual field names and values depend on the application's output schema.

## Important Notes

- Recognition accuracy depends on image quality, lighting, camera angle, motion blur, and plate visibility.
- OCR output may contain incorrect characters. Recognition results should be validated where accuracy is critical.
- Confidence scores indicate model confidence; they do not guarantee that a recognized plate is correct.
- Model weights and OCR assets may need to be obtained separately, depending on their licensing and repository configuration.
- Generated test results, temporary files, local environments, and sensitive configuration files should generally remain outside the public repository.
- Never commit passwords, API tokens, private camera credentials, or other secrets.

## Limitations

- Small, obstructed, damaged, or blurred license plates may be difficult to recognize.
- Performance varies with the input resolution, model size, hardware, and video frame rate.
- Plate formats and recognition rules may need configuration for different countries or regions.
- Real-world accuracy must be evaluated against correctly labelled test data.

## Future Improvements

Potential improvements include:

- Character-level OCR confidence reporting
- Multi-frame evidence aggregation
- Improved handling of uncertain recognition results
- Support for additional regional license plate formats
- Performance optimization for CPU-based inference
- Accuracy evaluation using labelled datasets

These are potential enhancements, not claims that every item is already implemented.

## License

This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0). See the [LICENSE](LICENSE) file for the full license terms.

Third-party libraries, detection models, OCR engines, and model weights may have separate license terms. Refer to their respective licenses for details.
