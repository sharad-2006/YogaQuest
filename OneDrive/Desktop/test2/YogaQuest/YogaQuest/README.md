# YogaQuest 🧘 - AI-Powered Yoga Pose Detection

A real-time yoga pose detection and scoring application using YOLO pose estimation and Streamlit.

## Features

- 📸 **Practice Mode** - Capture and analyze yoga poses
- 📹 **Live Session** - Real-time continuous pose tracking
- 🎬 **Video Analysis** - Upload and analyze recorded sessions
- 🧱 **Pose Wall Game** - Dynamic obstacle avoidance game
- 🎯 **Pose Target Game** - Hit specific pose targets
- 🌊 **Falling Pose Game** - Fast-paced pose-catching arcade game
- ☀️ **Surya Namaskar** - Guided Sun Salutation sequence
- 🏆 **Leaderboard** - Track progress and compete
- 👤 **Profile** - View achievements and statistics

## Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the App**
   ```bash
   double-click START_YOGAQUEST.bat
   ```
   Or manually:
   ```bash
   python -m streamlit run app.py
   ```

3. **Start Playing**
   - Enter your name
   - Choose a game mode
   - Strike yoga poses in front of your webcam!

## Technical Components

### Core Application
- `app.py` - Main Streamlit application with all game modes
- `database.py` - SQLite database for user stats and history

### Pose Detection & Analysis
- `pose_detector.py` - YOLO11-based pose detection
- `pose_analyzer.py` - Pose scoring and session analysis
- `poses_config.py` - Yoga pose definitions and configurations

### Camera & Performance
- `camera_manager.py` - Stable camera capture with background threading
- `frame_processor.py` - Async pose processing for smooth performance
- `app_camera_integration.py` - Integration layer for Streamlit

### Game Engines
- `game_engine.py` - XP, leveling, and achievement system
- `wall_game_engine.py` - Dynamic pose wall challenge
- `wall_renderer.py` - Wall game visualization
- `motion_analyzer.py` - Motion tracking and analysis

### Video Processing
- `video_processor.py` - Batch video analysis

## System Requirements

- Python 3.10+
- Webcam
- Windows / Linux / macOS
- 4GB RAM minimum

## Dependencies

- streamlit
- opencv-python
- numpy
- pandas
- ultralytics (YOLO)
- plotly

See `requirements.txt` for complete list.

## Performance

- **Smooth 20-30 FPS** UI updates
- **Stable camera connection** with no flickering
- **Async processing** for responsive experience
- **Background threading** for optimal performance

## Project Structure

```
files/
├── app.py                      # Main application
├── START_YOGAQUEST.bat         # Quick launcher
├── requirements.txt            # Dependencies
├── yogaquest.db               # User database
├── yolo11n-pose.pt            # YOLO model (nano)
├── yolo11s-pose.pt            # YOLO model (small)
├── .streamlit/                # Streamlit config
├── assets/                    # Pose reference images
├── camera_manager.py          # Camera handling
├── frame_processor.py         # Async processing
├── app_camera_integration.py  # Integration API
├── pose_detector.py           # YOLO pose detection
├── pose_analyzer.py           # Pose scoring
├── poses_config.py            # Pose definitions
├── database.py                # Data persistence
├── game_engine.py             # Game logic
├── wall_game_engine.py        # Wall game
├── wall_renderer.py           # Wall visualization
├── motion_analyzer.py         # Motion tracking
└── video_processor.py         # Video analysis
```

## Usage

1. **Practice Mode**: Capture individual poses for analysis
2. **Live Session**: Continuous real-time pose tracking
3. **Games**: Interactive pose-based mini-games
4. **Video Upload**: Analyze pre-recorded yoga sessions

## Credits

Built with:
- Streamlit for the web interface
- Ultralytics YOLO11 for pose detection
- OpenCV for video processing
- SQLite for data storage

---

**Enjoy your yoga practice! 🧘‍♀️**
