CN4242 Lab 3: Viral Item Prediction in Social Networks
Student: A0105772H

## Usage of tweet classifier
1. Open terminal (Ubuntu/Mac) or command line prompt (Windows).
2. Check the current version of Python. The predictor scripts are written in Python 3. Ensure that Python 3 is in use.
3. Redirect to the project folder: "viral-detection-of-micro-videos".
4. Run the following commands to prepare the textual video descriptions for predicting popularity.
python video_text_preprocessor.py, text_vader_sentiment_analyzer.py
5. Type "python" plus the script name of the predictor to make predictions. Press Enter key to execute the script.
e.g. "python visual_predictor.py",  "python late_fusion_predictor.py".
6. There are 5 analyzers available:
Level 0: "visual_predictor.py", "textual_predictor.py", "social_predictor.py"
Level 1: "late_fusion_predictor.py", "early_fusion_predictor.py"