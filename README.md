# 🛍️ RFM Customer Segmentation Dashboard

An interactive Streamlit dashboard that segments online retail customers using RFM analysis and KMeans clustering.

---

## 📌 Project Overview

This project uses **RFM (Recency, Frequency, Monetary)** analysis to segment customers based on their transaction behavior. Clustering is done using **KMeans**, and the results are visualized through an interactive **Streamlit** dashboard.


👨‍💻 Author
Sobikul Mia
📧 Email: sobikulmia11@gmail.com
🌐 GitHub: https://github.com/SobikulMia

🌐 Try the Live Demo:
🔗 https://online-retail-rfm-clustering-tzxkefczsrzumvtzosf5zq.streamlit.app/

## 📁 Project Files

| File Name                                      | Description                                      |
|-----------------------------------------------|--------------------------------------------------|
| `streamlit_retail_customer_segmentation.py`   | The main Streamlit app file to run the dashboard |
| `Online_Retail_Customer_Segmentation.ipynb`   | Jupyter notebook version of RFM and clustering   |
| `requirements.txt`                            | Required Python packages                         |
| `README.md`                                   | Project documentation file                       |

---

## 🚀 Technologies Used

- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- KMeans Clustering
- Streamlit
- Joblib

---

## 🧠 Features

- Upload Excel retail data
- Clean and preprocess dataset
- RFM feature engineering
- Optimal cluster finding using Elbow & Silhouette methods
- Visualize clusters and segment distribution
- Predict new customer cluster

---

## 🖥️ How to Run

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/rfm-customer-segmentation-dashboard.git
cd rfm-customer-segmentation-dashboard

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate     # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the Streamlit app
streamlit run streamlit_retail_customer_segmentation.py
