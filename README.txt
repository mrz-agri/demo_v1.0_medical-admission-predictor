 🎓 Medical Admission Predictor (Demo)

This repository provides a **demo version** of a machine learning model that predicts university admission outcomes for candidates in the **Konkur (Iranian university entrance exam, Experimental stream)**.

⚠️ **Note:**  
This demo uses a small synthetic dataset (30 samples).  
The real dataset and advanced preprocessing steps are private.  
On the real dataset, the model achieves around **90–93% accuracy**.

---

🛠 Requirements

- Python 3.8+
- pandas
- scikit-learn
- matplotlib
- imageio

You can install them easily:

```bash
pip install -r requirements.txt


How to Run
Clone this repository:

git clone https://github.com/mrz-agri/demo_v1.0_medical-admission-predictor.git
cd demo_v1.0_medical-admission-predictor


Run the demo script:
python demo2_uniadmission


This will:

Print accuracy scores and a classification report in the terminal

Generate an animated GIF (rf_cv_demo.gif) that shows probability predictions


Example Output

Accuracy on demo dataset: 100% (⚠️ synthetic dataset, very small)

Accuracy on real dataset: 90–93% (private)

Output includes:

Per-fold accuracy scores

Classification report

Animated GIF


Note on Privacy

The full dataset and complete preprocessing code are kept private due to sensitivity and confidentiality.
This repository is only for demonstration and educational purposes