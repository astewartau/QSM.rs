#!/bin/bash
# Download test data from OSF project y8adf
# Requires OSF_TOKEN environment variable for private project access

set -e

cd "$(dirname "$0")/.."

if [ -d "bids" ]; then
    echo "bids/ directory already exists. Remove it first if you want to re-download."
    exit 0
fi

if [ -z "$OSF_TOKEN" ]; then
    echo "Error: OSF_TOKEN environment variable not set"
    echo "Get a token from https://osf.io/settings/tokens"
    exit 1
fi

echo "Downloading test data from OSF project y8adf..."
curl -L -H "Authorization: Bearer ${OSF_TOKEN}" \
    "https://files.osf.io/v1/resources/y8adf/providers/osfstorage/698ac9aecae88916d1e24f69" \
    -o bids.zip

echo "Extracting..."
unzip -q bids.zip
rm bids.zip

echo "Done. Test data is in bids/"
ls -la bids/
