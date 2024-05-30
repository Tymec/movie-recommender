"""Entry point for Hugging Face gradio space."""

from app.gui import launch_gui

launch_gui("models/sentiment140_tfidf_ft-20000.pkl", share=False)
