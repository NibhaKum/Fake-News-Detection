# AI-Powered Fake News Detection

A text-classification project using TF-IDF and four machine-learning algorithms:
KNN, Logistic Regression, Random Forest and MLP Neural Network.

The Streamlit application uses the Random Forest model selected from the
experiment in the accompanying notebook.

## Repository files

- `2_FakeNews_SourceCode_Deployment_Ready.ipynb` — training, evaluation and model export.
- `app.py` — Streamlit web application.
- `requirements.txt` — Python dependencies.
- `model/fake_news_model.pkl` — generated after running the notebook export cell.
- `model/tfidf_vectorizer.pkl` — generated after running the notebook export cell.

## Before deploying

1. Open the notebook in Google Colab or Jupyter.
2. Upload `True.csv` and `Fake.csv` to the notebook runtime.
3. Run the notebook cells in order.
4. Run the added **SAVE THE TRAINED MODEL AND TF-IDF VECTORIZER FOR DEPLOYMENT** cell.
5. Download the two generated files from the `model/` folder.
6. Create a `model` folder in this repository and upload both `.pkl` files into it.
7. Upload `app.py`, `requirements.txt`, and the notebook.

## Deploy on Streamlit Community Cloud

1. Sign in at https://share.streamlit.io/ using GitHub.
2. Select **Create app** and choose this repository.
3. Select branch `main`.
4. Set the main file path to `app.py`.
5. Deploy.

## Important model compatibility note

The `.pkl` model and vectorizer must be loaded with a compatible scikit-learn
version. The notebook prints the version used when the artifacts are created.
If deployment reports a model-version warning or loading error, pin that exact
version in `requirements.txt`, for example `scikit-learn==X.Y.Z`, replacing
`X.Y.Z` with the version printed by your notebook.

## Class labels

- `0` = Fake News
- `1` = Real News

This is a machine-learning text classifier, not a fact-checking service.
Its output should not be treated as proof that an article is true or false.
