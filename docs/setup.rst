System Setup
============

Bionium-X uses a standard Python stack initialized within a virtual environment.

Dependencies and Environments
-----------------------------

The platform is optimized for Unix/macOS environments but remains highly portable. For optimal deep learning inference, `PyTorch` CPU fallback is supported although GPU acceleration (CUDA) is recommended where active model retraining is necessary.

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/organization/bionium-x.git
   cd bionium-x

   # Set up virtual environment
   python -m venv venv
   source venv/bin/activate

   # Install strictly-versioned dependencies
   pip install -r requirements.txt

Running the Application
-----------------------

Initialization of the primary application is handled via `Streamlit`:

.. code-block:: bash

   streamlit run app.py

The application will default to running on `localhost:8501`.
