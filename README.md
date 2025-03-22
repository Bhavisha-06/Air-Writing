## **Air-Writing** : Hand Writing Recognition in Air with Computer Vision

This project implements AirWrite, a system that combines hand tracking with Google Cloud Vision to recognize text written in the air with your finger. The system uses your webcam to track hand movements, detects when you're "writing" with your index finger, and then uses Google's Vision API to convert your air-written gestures into text.

## Research Paper

This implementation is based on research in the field of gesture-based interfaces and computer vision. For more detailed information about the methodology, algorithms, and evaluation, please refer to our research paper:

[MODEL IMPLEMENTATION AND COMPREHENSIVE STUDY ON VISUAL AIR TYPING SYSTEMS] - [Bhavisha Narendra Chaudhari
, Adarsh Jha
] - [International Journal of Engineering Applied Sciences and Technology] - [2024]
[https://www.ijeast.com/papers/95-99,%20Tesma0904,IJEAST.pdf]

## Features

- Real-time hand tracking using MediaPipe
- Gesture-based writing detection using finger angle analysis
- Automatic text recognition when pausing between writing
- Simple drawing interface showing your writing strokes

## Requirements

- Python 3.6+
- OpenCV
- MediaPipe
- NumPy
- Google Cloud Vision API
- A webcam

## Installation

1. Clone this repository:
```bash
git clone https://github.com/Bhavisha-06/Air-Writing.git
cd airwrite
```

2. Install the required packages:
```bash
pip install opencv-python mediapipe numpy google-cloud-vision
```

3. Set up Google Cloud Vision API:
   - Create a project in the [Google Cloud Console](https://console.cloud.google.com/)
   - Enable the Vision API for your project
   - Create a service account and download the JSON key file
   - Keep the JSON key file secure and do not share it publicly

## Usage

Run the script with your Google Cloud API key file:

```bash
python airwrite.py --api_key path/to/your-api-key.json
```

### How to Use:

1. Point your index finger at the camera
2. Extend your index finger straight (finger angle > 160°) to start "writing"
3. Move your finger in the air to write letters or words
4. Pause for 5 seconds to trigger text recognition
5. The recognized text will appear in the console
6. Press 'q' to quit the application

## How It Works

1. **Hand Tracking**: MediaPipe's hand tracking model identifies hand landmarks in the webcam feed
2. **Gesture Detection**: The system calculates the angle of your index finger to determine when you're writing
3. **Stroke Capturing**: When writing is detected, the system records the path of your fingertip
4. **Pause Detection**: After 5 seconds of inactivity, the system processes your writing
5. **Text Recognition**: Google Cloud Vision API analyzes the drawing to recognize the text
6. **Results**: Recognized text is displayed in the console

## Customization

You can adjust several parameters in the code:
- `WRITING_ANGLE_THRESHOLD`: Change the angle threshold for detecting writing (default: 160°)
- `PAUSE_THRESHOLD`: Adjust the waiting time before recognizing text (default: 5 seconds)
- Modify the hand detection confidence threshold in the `mp_hands.Hands()` constructor

## Limitations

- Currently supports single-hand tracking only
- Recognition accuracy depends on clear, deliberate movements
- Requires a stable internet connection for API communication

## Citation

If you use AirWrite in your research or project, please cite our paper:

```
@article{jha2024model,
  title={MODEL IMPLEMENTATION AND COMPREHENSIVE STUDY ON VISUAL AIR TYPING SYSTEMS},
  author={Jha, Adarsh and Chaudhari, Bhavisha Narendra},
  journal={International Journal of Engineering Applied Sciences and Technology},
  volume={09},
  number={04},
  pages={95-99},
  year={2024},
  month={September},
  doi={10.33564/IJEAST.2024.v09i04.012},
  publisher={IJEAST}
}
```

## License

[MIT License](LICENSE)

## Acknowledgments

- [MediaPipe](https://google.github.io/mediapipe/) for the hand tracking solution
- [Google Cloud Vision API](https://cloud.google.com/vision) for text recognition capabilities
- [OpenCV](https://opencv.org/) for image processing
