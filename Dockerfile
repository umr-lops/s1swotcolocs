# ---- Stage 1: Micromamba Base ----
# This stage just installs micromamba for reuse.
FROM ubuntu:22.04 as micromamba-base

ARG MAMBA_VERSION=1.5.6
ENV MAMBA_ROOT_PREFIX=/opt/conda

RUN apt-get update && apt-get install -y curl bzip2 ca-certificates && \
    curl -L https://micromamba.snakepit.net/api/micromamba/linux-64/${MAMBA_VERSION} | \
    tar -xvj --strip-components=1 -C /usr/local/bin/ bin/micromamba && \
    apt-get clean && rm -rf /var/lib/apt/lists/*


# ---- Stage 2: The Builder ----
FROM micromamba-base as builder

# Receive the CI_JOB_TOKEN directly
ARG CI_JOB_TOKEN

# Copy your environment file
COPY environment.yml /tmp/environment.yml

# Create the base environment
RUN micromamba create -y -n myenv -f /tmp/environment.yml && \
    micromamba clean -a -y

# Install s1ifr using the registry URL
# We use --no-cache-dir to keep the image small
# We use --extra-index-url to allow pip to look at standard PyPI AND your private GitLab
RUN micromamba run -n myenv pip install s1ifr \
    --no-cache-dir \
    --extra-index-url https://__token__:${CI_JOB_TOKEN}@gitlab.ifremer.fr/api/v4/projects/4991/packages/pypi/simple
    # --extra-index-url https://gitlab-ci-token:${CI_JOB_TOKEN}@gitlab.ifremer.fr/api/v4/projects/4991/packages/pypi/simple



# ---- Stage 3: The Final Image ----
# This is the image you will actually use. It is clean and never saw the secret.
FROM ubuntu:22.04

# Set up environment variables for the final image
ENV MAMBA_ROOT_PREFIX=/opt/conda \
    PATH=/opt/conda/bin:$PATH \
    ENV_NAME=myenv

# Install only the RUNTIME system dependencies. No need for git, curl, etc.
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libxext6 libsm6 libxrender1 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# --- THE MAGIC STEP ---
# Copy the entire conda environment, with s1ifr already installed, from the builder stage.
COPY --from=builder /opt/conda /opt/conda

# Set up the shell to use the activated environment by default
ENV PATH=$MAMBA_ROOT_PREFIX/envs/$ENV_NAME/bin:$PATH

# Set the working directory
WORKDIR /app

# Copy the application source code
COPY . /app

# Install your local application code into the environment
# The environment already contains s1ifr and all its dependencies
RUN pip install .

# Set the default command for the container
CMD ["python"]
