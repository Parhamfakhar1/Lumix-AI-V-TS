#!/bin/bash
# scripts/deploy.sh

VERSION=${1:-"1.0.0"}
PLATFORMS=("linux/amd64" "linux/arm" "linux/arm64" "windows/amd64")

echo "📦 Packaging Lumix AI V-TS v$VERSION"

# ساخت برای تمام پلتفرم‌ها
for platform in "${PLATFORMS[@]}"; do
    OS=${platform%/*}
    ARCH=${platform#*/}
    
    echo "🔨 Building for $OS/$ARCH..."
    
    OUTPUT_NAME="lumix-ai-vts-$VERSION-$OS-$ARCH"
    if [ "$OS" = "windows" ]; then
        OUTPUT_NAME="$OUTPUT_NAME.exe"
    fi
    
    GOOS=$OS GOARCH=$ARCH go build \
        -ldflags="-s -w -X main.version=$VERSION" \
        -o "dist/$OUTPUT_NAME" \
        ./cmd/lumix
    
    # فشرده‌سازی
    if command -v upx &> /dev/null; then
        echo "📦 Compressing with UPX..."
        upx --best "dist/$OUTPUT_NAME"
    fi
done

# ایجاد پکیج
echo "📁 Creating distribution packages..."

# لینوکس
tar -czf "dist/lumix-ai-vts-$VERSION-linux.tar.gz" \
    -C dist \
    lumix-ai-vts-$VERSION-linux-amd64 \
    lumix-ai-vts-$VERSION-linux-arm \
    lumix-ai-vts-$VERSION-linux-arm64

# ویندوز
zip -j "dist/lumix-ai-vts-$VERSION-windows.zip" \
    dist/lumix-ai-vts-$VERSION-windows-amd64.exe

# ایجاد checksum
echo "🔐 Generating checksums..."
cd dist && sha256sum * > SHA256SUMS && cd ..

echo "✅ Deployment packages created in dist/"
ls -lh dist/