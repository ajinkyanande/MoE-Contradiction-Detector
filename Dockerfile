# Use the official Miniconda image as a base
FROM continuumio/miniconda3

# Set the working directory
WORKDIR /app

# Copy the environment file
COPY environment.yml .

# Create the conda environment and clean up
RUN conda env create -n moe-cd -f environment.yml && conda clean -afy

# Update PATH so that the new conda environment is used
ENV PATH="/opt/conda/envs/moe-cd/bin:$PATH"

# Copy your application code
COPY . .

# Expose port 8080 for SageMaker
EXPOSE 8080

# Start the application
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8080"]
