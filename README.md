# License Plate Recognition

Detects and reads vehicle license plates in a video using a fine-tuned YOLO
model for plate detection and EasyOCR for character recognition. Includes
format validation for Indian plates (`AA00AAA`), OCR error correction
(e.g. `0`↔`O`, `1`↔`I`), and a majority-vote buffer that stabilises the read
text across frames.

## Files

- `License_plate_detection.py` — main inference script
- `License_plate_detector.ipynb` — notebook version
- `license_plate_best.pt` — fine-tuned YOLO detection model (~6 MB)
- `requirements.txt`, `.gitignore`, `README.md`

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

On the **first run**, EasyOCR downloads its detection and recognition models
(~100 MB) to a local cache — this needs an internet connection and happens
only once.

## Run

1. Put your input video in the project folder as `video.mp4`
   (or edit `input_video` at the bottom of the script).
2. Run:
   ```bash
   python License_plate_detection.py
   ```
3. A preview window opens; press `q` to stop early. The annotated result is
   written to `output_video.mp4`.

## Notes

- The script uses `easyocr.Reader(['en'], gpu=True)`. On a machine without a
  CUDA GPU, EasyOCR prints a notice and falls back to CPU automatically
  (slower, but works).
- `requirements.txt` pins the full `opencv-python` because this script opens a
  preview window (`cv2.imshow`). If you ever deploy it to a headless server or
  Streamlit, switch to `opencv-python-headless` **and** remove the
  `cv2.imshow` / `cv2.waitKey` / `cv2.destroyAllWindows` calls first — those
  GUI functions are not usable in the headless build.
- Input/output videos are gitignored so large media doesn't bloat the repo.
