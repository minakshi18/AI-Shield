# AI Shield - Advanced Fake News Detection

An AI-powered web application that detects fake news using machine learning and natural language processing.

## Features

- **Instant Detection**: Analyze news articles in real-time
- **High Accuracy**: 94.2% model accuracy trained on 40,000+ verified news records
- **Multi-format Input**: Paste text or upload news screenshots for OCR analysis
- **Confidence Scores**: Get detailed Real vs Fake breakdown with percentage confidence
- **Fast Processing**: Results in less than 0.5 seconds

## Technologies

- **Backend**: Python, Flask
- **ML/NLP**: Scikit-Learn, NLTK
- **OCR**: Tesseract OCR
- **Frontend**: HTML, CSS, JavaScript
- **UI Framework**: FontAwesome Icons

## Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

1. Clone the repository
```bash
git clone https://github.com/yourusername/FakeNews_Project.git
cd FakeNews_Project
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Run the application
```bash
python app.py
```

4. Open your browser and navigate to
```
http://localhost:5000
```

## Model Performance

| Metric | Value |
|--------|-------|
| Accuracy | 94.2% |
| Precision | 92.1% |
| Dataset Size | 40,000+ records |
| Response Time | 0.2 seconds |

## Project Structure

```
FakeNews_Project/
├── app.py                 # Flask application
├── model_training.py      # Model training script
├── data_preparation.py    # Data preprocessing
├── preprocessing.py       # Text preprocessing utilities
├── vectorization.py       # Feature extraction
├── eda_analysis.py        # Exploratory data analysis
├── requirements.txt       # Python dependencies
├── data/                  # Dataset files
│   ├── Fake.csv
│   ├── True.csv
│   ├── cleaned_data.csv
│   └── processed_data.csv
├── models/                # Trained ML models
├── static/                # CSS & frontend assets
│   └── style.css
└── templates/             # HTML templates
    └── index.html
```

## Usage

1. **Text Analysis**: Paste any news article or headline in the text area
2. **Image Analysis**: Upload a screenshot of news for OCR extraction and analysis
3. **Get Results**: Receive Real/Fake prediction with confidence percentage

## Developers

- Meenakshi Neolia
- Parul Mishra
- Priya Singh
- Vyom Dixit

## License

© 2026 All Rights Reserved

## Contributing

Feel free to fork this project and submit pull requests for improvements.

---

**Built with ❤️ for fake news detection**
