#!/bin/bash
# Generate LocalData MCP Server logo
# This script creates a properly sized rectangular logo with rounded corners
# Logo is not stored in git - generated locally as needed

set -e

echo "🎨 Generating LocalData MCP Server logo..."

# Check if ImageMagick is available
if ! command -v magick &> /dev/null; then
    echo "❌ ImageMagick is required but not installed."
    echo "   Install with: brew install imagemagick"
    exit 1
fi

# Define source - this should be the high-quality icon2.png
SOURCE_ICON="/Users/chris/Downloads/icon2.png"

if [ ! -f "$SOURCE_ICON" ]; then
    echo "❌ Source icon not found at: $SOURCE_ICON"
    echo "   Please ensure icon2.png is in ~/Downloads/"
    exit 1
fi

echo "📁 Using source: $SOURCE_ICON"

# Create logo directory if it doesn't exist
mkdir -p assets/logos

echo "🔧 Creating rectangular logo with proper dimensions..."

# Create rectangular version with trimmed white space and proper aspect ratio
magick "$SOURCE_ICON" \
    -trim +repage \
    -resize 240x80^ \
    -gravity center \
    -background transparent \
    -extent 240x80 \
    assets/logos/logo_rectangular.png

# Create rounded corner version
echo "🔄 Adding proper rounded corners..."
magick assets/logos/logo_rectangular.png \
    \( +clone -alpha extract \
       -draw 'fill black polygon 0,0 0,15 15,0 fill white circle 15,15 15,0' \
       \( +clone -flip \) -compose Multiply -composite \
       \( +clone -flop \) -compose Multiply -composite \
    \) \
    -alpha off -compose CopyOpacity -composite logo.png

# Create different sizes for various uses
echo "📏 Creating different sizes..."

# GitHub README size (72x36 - 2:1 aspect ratio)
magick logo.png -resize 144x72 assets/logos/logo_readme.png

# Icon sizes (square versions)
for size in 16 32 64 128 256; do
    magick "$SOURCE_ICON" -resize ${size}x${size} assets/logos/logo_${size}.png
    echo "   ✅ Created ${size}x${size}"
done

# Create favicon
magick "$SOURCE_ICON" -resize 32x32 favicon.ico

echo "📊 Generated logos:"
ls -la logo.png favicon.ico assets/logos/ 2>/dev/null || true

echo ""
echo "✅ Logo generation complete!"
echo "📝 Usage for README:"
echo "   <img src=\"logo.png\" alt=\"LocalData MCP Server\" width=\"72\" height=\"36\">"
echo ""
echo "💡 Logo files are excluded from git via .gitignore"
echo "🔧 Run this script anytime to regenerate logo assets"