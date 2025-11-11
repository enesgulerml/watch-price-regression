# -----
# V4 (API) Konteyner Tarifi
# -----

# Step 1: Base Image
FROM continuumio/miniconda3:latest

# Step 2: Working Directory
WORKDIR /app

# Step 3: Copy Dependencies
COPY environment.yml .

# Step 4: Install Conda Environment (watch-ml)
RUN conda env create -f environment.yml

# Step 5: Introduce the conda environment to the 'bash' shell
SHELL ["/bin/bash", "-c"]
RUN echo "source activate watch-ml" > ~/.bashrc
ENV PATH="/opt/conda/envs/watch-ml/bin:$PATH"

# Step 6: Copy Project Codes
COPY . .

# Step 7: Install the Package Inside the Container
RUN pip install -e .

# Step 8: Expose Port
EXPOSE 8000

# Step 9: Start the Server (CMD)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]