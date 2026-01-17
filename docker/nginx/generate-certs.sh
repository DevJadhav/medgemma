#!/bin/bash
# Generate self-signed SSL certificates for development
# For production, use Let's Encrypt with certbot

set -e

CERT_DIR="/etc/nginx/ssl"
DAYS_VALID=365

# Create directory if it doesn't exist
mkdir -p "$CERT_DIR"

# Check if certificates already exist
if [ -f "$CERT_DIR/cert.pem" ] && [ -f "$CERT_DIR/key.pem" ]; then
    echo "Certificates already exist. Skipping generation."
    echo "To regenerate, delete the existing certificates first."
    exit 0
fi

echo "Generating self-signed SSL certificates..."

# Generate private key and certificate
openssl req -x509 \
    -nodes \
    -days "$DAYS_VALID" \
    -newkey rsa:2048 \
    -keyout "$CERT_DIR/key.pem" \
    -out "$CERT_DIR/cert.pem" \
    -subj "/C=US/ST=California/L=San Francisco/O=MedAI Compass/OU=Development/CN=localhost" \
    -addext "subjectAltName=DNS:localhost,DNS:*.localhost,IP:127.0.0.1"

# Set proper permissions
chmod 600 "$CERT_DIR/key.pem"
chmod 644 "$CERT_DIR/cert.pem"

echo "Self-signed certificates generated successfully!"
echo "  Certificate: $CERT_DIR/cert.pem"
echo "  Private Key: $CERT_DIR/key.pem"
echo ""
echo "NOTE: These are self-signed certificates for DEVELOPMENT only."
echo "For production, use Let's Encrypt or a trusted CA."
