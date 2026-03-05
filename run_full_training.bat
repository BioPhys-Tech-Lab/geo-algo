@echo off
echo Starting Full Model Training...
echo This process may take several hours on CPU.
echo Press Ctrl+C to cancel at any time.
python src/classifier.py --epochs 10
echo Training Complete! Model saved to rock_classifier.pth
pause
