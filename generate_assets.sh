#!/bin/bash
# Generate logo assets for LocalData MCP Server
# This script creates various icon sizes without storing them in git

set -e

echo "🎨 Generating LocalData MCP Server assets..."

# Check if ImageMagick is available
if ! command -v magick &> /dev/null; then
    echo "❌ ImageMagick is required but not installed."
    echo "   Install with: brew install imagemagick"
    exit 1
fi

# Check if logo.png exists
if [ ! -f "logo.png" ]; then
    echo "❌ logo.png not found in current directory"
    exit 1
fi

# Create assets directory
mkdir -p assets/icons

echo "📁 Creating icon sizes..."

# Generate different sizes
for size in 16 32 64 128 256 512; do
    magick logo.png -resize ${size}x${size} assets/icons/logo_${size}.png
    echo "   ✅ Created ${size}x${size}"
done

# Create favicon
echo "🌐 Creating favicon..."
magick logo.png -resize 32x32 favicon.ico

# Create rounded version (optional)
echo "🔄 Creating rounded version..."
magick logo.png \
    \( +clone -alpha extract -draw "fill black polygon 0,0 0,20 20,0" \
       \( +clone -flip \) -compose Multiply -composite \
       \( +clone -flop \) -compose Multiply -composite \) \
    -alpha off -compose CopyOpacity -composite assets/icons/logo_rounded.png

echo "📊 Asset summary:"
ls -la assets/icons/ | grep logo
ls -la favicon.ico 2>/dev/null || echo "   favicon.ico created"

echo ""
echo "✅ All assets generated successfully!"
echo "💡 Assets are excluded from git via .gitignore"
echo "🔧 Run this script anytime to regenerate assets"