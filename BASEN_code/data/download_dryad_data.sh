#!/bin/bash

echo "Starting Dryad dataset download..."

# Cocktail Party.zip
wget "https://dryad-assetstore-merritt-west.s3.us-west-2.amazonaws.com/ark%3A/13030/m5c87bqr%7C1%7Cproducer/Cocktail%20Party.zip?response-content-disposition=attachment%3B%20filename%3DCocktail%2BParty.zip&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIA2KERHV5E3OITXZXC%2F20250614%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20250614T191902Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=6cb4dc9be4568bed8357a270c063513fb806643e6c16eec2268fbde70450ba53" -O "Cocktail_Party.zip"

# N400.zip
wget "https://dryad-assetstore-merritt-west.s3.us-west-2.amazonaws.com/ark%3A/13030/m5c87bqr%7C1%7Cproducer/N400.zip?response-content-disposition=attachment%3B%20filename%3DN400.zip&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIA2KERHV5E3OITXZXC%2F20250614%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20250614T191902Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=9f4e253b846bd2b115b17a6313f49f3d7e72841897a52cc488af85cd05ef1691" -O "N400.zip"

# Natural Speech - Reverse.zip
wget "https://dryad-assetstore-merritt-west.s3.us-west-2.amazonaws.com/ark%3A/13030/m5c87bqr%7C1%7Cproducer/Natural%20Speech%20-%20Reverse.zip?response-content-disposition=attachment%3B%20filename%3DNatural%2BSpeech%2B-%2BReverse.zip&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIA2KERHV5E3OITXZXC%2F20250614%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20250614T191902Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=91113bdf210c28fcdc75487a54fe3d1f4ce60c07effd5296b5150ea6a07aabfc" -O "Natural_Speech_Reverse.zip"

# Natural Speech.zip
wget "https://dryad-assetstore-merritt-west.s3.us-west-2.amazonaws.com/ark%3A/13030/m5c87bqr%7C1%7Cproducer/Natural%20Speech.zip?response-content-disposition=attachment%3B%20filename%3DNatural%2BSpeech.zip&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIA2KERHV5E3OITXZXC%2F20250614%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20250614T191902Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=9390f3abf6eb7e0159d10d340de9b91446e422a05c1fa4a3b045136fbbc53dee" -O "Natural_Speech.zip"

echo "✅ Done."
