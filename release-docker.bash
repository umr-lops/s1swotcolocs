#!/bin/bash

# Prompt the user for the version
read -p "Enter the version (e.g., 0.0.2): " VERSION
#read -p "Enter the gitlab token to access private repo https://gitlab/lops-wave/s1ifr/: " TOKENGITLAB
# Check if the version is provided
if [ -z "$VERSION" ]; then
  echo "Error: Version not provided."
  exit 1
fi

# Define image names
BASE_IMAGE_NAME="s1swot"
REGISTRY_IMAGE_NAME="gitlab-registry.ifremer.fr/lops-wave/s1ifr:s1swot"

# get CI_JOB_TOKEN from environment variable
CI_JOB_TOKEN=${CI_JOB_TOKEN:-}
# test if CI_JOB_TOKEN is set and not empty
if [ -n "$CI_JOB_TOKEN" ]; then
    echo "Variable CI_JOB_TOKEN is set and not empty."
else
    echo "Variable CI_JOB_TOKEN is not set or is empty."
fi

# if [ -n "$TOKENGITLAB" ]; then
#     echo "Variable TOKENGITLAB is set and not empty."
# else
#     echo "Variable TOKENGITLAB is not set or is empty."
# fi
# Docker commands
echo "Building Docker image: ${BASE_IMAGE_NAME}:${VERSION}"
#docker build . --no-cache --build-arg GITLAB_CREDS=$(cat gitlab_creds) -t "${BASE_IMAGE_NAME}:${VERSION}"
docker build . --no-cache --build-arg GITLAB_CREDS=$(cat gitlab_creds) --build-arg CI_JOB_TOKEN=$CI_JOB_TOKEN -t "${BASE_IMAGE_NAME}:${VERSION}"

echo "Tagging Docker image: ${BASE_IMAGE_NAME}:${VERSION} -> ${REGISTRY_IMAGE_NAME}-${VERSION}"
docker tag "${BASE_IMAGE_NAME}:${VERSION}" "${REGISTRY_IMAGE_NAME}-${VERSION}"

echo "Pushing Docker image: ${REGISTRY_IMAGE_NAME}-${VERSION}"
docker push "${REGISTRY_IMAGE_NAME}-${VERSION}"

echo "Release process completed for version ${VERSION}."
