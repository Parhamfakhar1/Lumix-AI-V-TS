#!/bin/bash
# scripts/setup.sh

set -e

echo "🚀 Lumix AI V-TS Setup Script"
echo "=============================="

# بررسی وجود Go
if ! command -v go &> /dev/null; then
    echo "❌ Go is not installed. Please install Go 1.21 or later."
    exit 1
fi

GO_VERSION=$(go version | awk '{print $3}' | sed 's/go//')
REQUIRED_VERSION="1.21"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$GO_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "❌ Go version $GO_VERSION is too old. Required: $REQUIRED_VERSION+"
    exit 1
fi

echo "✅ Go $GO_VERSION is installed"

# بررسی وجود وابستگی‌ها
echo "📦 Checking dependencies..."

REQUIRED_CMDS=("git" "make" "curl" "tar")
for cmd in "${REQUIRED_CMDS[@]}"; do
    if ! command -v "$cmd" &> /dev/null; then
        echo "❌ $cmd is not installed"
        exit 1
    fi
done

echo "✅ All dependencies are installed"

# ایجاد دایرکتوری‌ها
echo "📁 Creating directory structure..."

mkdir -p data/{training,models,config,archive,logs,knowledge}
mkdir -p bin
mkdir -p tests/{unit,integration}

echo "✅ Directories created"

# کپی فایل‌های پیکربندی
echo "⚙️  Copying configuration files..."

if [ -f "config/default.yaml" ]; then
    cp config/default.yaml data/config/
else
    echo "⚠️  Warning: config/default.yaml not found, creating default..."
    
    cat > data/config/default.yaml << 'EOF'
system:
  name: "Lumix AI V-TS"
  version: "1.0.0"
  mode: "development"
  debug: true

model:
  vocab_size: 8192
  hidden_size: 128
  num_layers: 4
  num_heads: 4
  max_seq_length: 256
  dropout: 0.1
  learning_rate: 0.001
  batch_size: 8
  warmup_steps: 1000
  weight_decay: 0.01

search:
  google_api_key: "${GOOGLE_API_KEY}"
  search_engine_id: "${SEARCH_ENGINE_ID}"
  max_results: 10
  query_variations: 9
  timeout: 10
  retry_attempts: 3
  cache_ttl: 24h

performance:
  max_goroutines: 4
  memory_limit_mb: 200
  cpu_cores: 2

logging:
  level: "info"
  format: "console"
EOF
fi

echo "✅ Configuration files copied"

# دانلود داده‌های آموزشی اولیه
echo "📥 Downloading training data..."

if [ ! -f "data/training/base_knowledge.jsonl" ]; then
    echo "Creating initial training data..."
    
    cat > data/training/base_knowledge.jsonl << 'EOF'
{"input": "سلام", "output": "سلام! چطور می‌تونم کمکتون کنم؟", "category": "greeting"}
{"input": "سلام چطوری؟", "output": "سلام! خوبم ممنون، شما چطورید؟", "category": "greeting"}
{"input": "خداحافظ", "output": "خداحافظ! روز خوبی داشته باشید.", "category": "greeting"}
{"input": "متشکرم", "output": "خواهش می‌کنم! خوشحالم که تونستم کمک کنم.", "category": "greeting"}
{"input": "لطفا", "output": "حتماً، با کمال میل.", "category": "greeting"}
{"input": "ببخشید", "output": "اشکالی نداره، بفرمایید.", "category": "greeting"}
{"input": "اسمت چیه؟", "output": "من Lumix AI V-TS هستم، یک دستیار هوشمند.", "category": "identity"}
{"input": "چه کاری می‌تونی انجام بدی؟", "output": "می‌تونم سوالات شما رو جواب بدم، در اینترنت جستجو کنم و از صحبت هامون یاد بگیرم.", "category": "capability"}
{"input": "چطور کار می‌کنی؟", "output": "من از هوش مصنوعی و جستجوی اینترنتی استفاده می‌کنم تا بهترین پاسخ رو بهتون بدم.", "category": "capability"}
{"input": "کجا زندگی می‌کنی؟", "output": "من یک برنامه کامپیوتری هستم و روی سرور یا کامپیوتر شما اجرا می‌شم.", "category": "identity"}
EOF
    
    echo "Generated 10 sample training entries"
else
    echo "✅ Training data already exists"
fi

# تنظیم متغیرهای محیطی
echo "🔧 Setting up environment variables..."

if [ ! -f ".env" ]; then
    cat > .env << 'EOF'
# Google Custom Search API
GOOGLE_API_KEY=your_api_key_here
SEARCH_ENGINE_ID=your_search_engine_id_here

# Application Settings
LUMIX_MODE=development
LUMIX_PORT=8080
LUMIX_OFFLINE_MODE=false

# Performance Settings
MAX_MEMORY_MB=200
MAX_CPU_CORES=2
MAX_GOROUTINES=4

# Logging
LOG_LEVEL=info
LOG_FORMAT=console
EOF
    
    echo "⚠️  Please edit .env file with your actual API keys"
fi

echo "✅ Environment variables set"

# نصب وابستگی‌های Go
echo "📦 Installing Go dependencies..."

go mod download
go mod verify

echo "✅ Dependencies installed"

# کامپایل اولیه
echo "🔨 Building project..."

make build-linux

if [ $? -eq 0 ]; then
    echo "✅ Build successful"
    echo ""
    echo "📊 Build Info:"
    ls -lh bin/
else
    echo "❌ Build failed"
    exit 1
fi

# آموزش اولیه مدل
read -p "🎓 Do you want to train the initial model? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Starting initial training..."
    make train
fi

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. Edit .env file with your Google API keys"
echo "2. Run: make run (for development)"
echo "3. Run: make run-offline (for offline mode)"
echo "4. Visit: http://localhost:8080"
echo ""
echo "For Raspberry Pi deployment:"
echo "make deploy-pi"
echo ""