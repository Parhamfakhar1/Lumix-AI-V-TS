# Makefile
.PHONY: all build test clean deploy setup train run

# تنظیمات پروژه
APP_NAME := lumix-ai-vts
VERSION := 1.0.0
BUILD_DIR := ./bin
DATA_DIR := ./data

# کامپایلر و فلگ‌ها
GO := go
GOFLAGS := -ldflags="-s -w -X main.version=$(VERSION)"
GOBUILD := $(GO) build $(GOFLAGS)

# اهداف پیش‌فرض
all: setup build

# راه‌اندازی اولیه پروژه
setup:
	@echo "📦 Setting up Lumix AI V-TS..."
	
	# ایجاد دایرکتوری‌ها
	mkdir -p $(BUILD_DIR)
	mkdir -p $(DATA_DIR)/{training,models,config,archive,logs,knowledge}
	
	# کپی فایل‌های پیکربندی
	cp config/default.yaml $(DATA_DIR)/config/
	
	# دانلود داده‌های آموزشی اولیه
	@if [ ! -f "$(DATA_DIR)/training/base_knowledge.jsonl" ]; then \
		echo "📥 Downloading training data..."; \
		curl -sL https://example.com/base_knowledge.jsonl -o $(DATA_DIR)/training/base_knowledge.jsonl; \
	fi
	
	@echo "✅ Setup completed!"

# کامپایل برای معماری‌های مختلف
build: build-linux build-arm build-windows

build-linux:
	@echo "🔨 Building for Linux..."
	GOOS=linux GOARCH=amd64 $(GOBUILD) -o $(BUILD_DIR)/$(APP_NAME)-linux-amd64 ./cmd/lumix

build-arm:
	@echo "🔨 Building for ARM (Raspberry Pi)..."
	GOOS=linux GOARCH=arm GOARM=5 $(GOBUILD) -o $(BUILD_DIR)/$(APP_NAME)-linux-armv5 ./cmd/lumix
	GOOS=linux GOARCH=arm64 $(GOBUILD) -o $(BUILD_DIR)/$(APP_NAME)-linux-arm64 ./cmd/lumix

build-windows:
	@echo "🔨 Building for Windows..."
	GOOS=windows GOARCH=amd64 $(GOBUILD) -o $(BUILD_DIR)/$(APP_NAME)-windows-amd64.exe ./cmd/lumix

# اجرای تست‌ها
test:
	@echo "🧪 Running tests..."
	$(GO) test ./... -v -cover -race -timeout 5m

# اجرای تست‌های یکپارچه‌سازی
test-integration:
	@echo "🧪 Running integration tests..."
	$(GO) test ./tests/integration -v -timeout 10m

# پاک‌سازی فایل‌های build
clean:
	@echo "🧹 Cleaning up..."
	rm -rf $(BUILD_DIR)
	rm -rf coverage.out
	rm -rf $(DATA_DIR)/archive/*.tmp
	rm -rf logs/*.log

# آموزش مدل اولیه
train:
	@echo "🎓 Training initial model..."
	$(GO) run ./cmd/lumix --train --epochs=3 --data=$(DATA_DIR)/training/

# اجرای توسعه
run:
	@echo "🚀 Starting development server..."
	$(GO) run ./cmd/lumix --config=$(DATA_DIR)/config/default.yaml --verbose

# اجرای حالت آفلاین
run-offline:
	@echo "📴 Starting in offline mode..."
	$(GO) run ./cmd/lumix --offline --config=$(DATA_DIR)/config/default.yaml

# بررسی کیفیت کد
lint:
	@echo "🔍 Linting code..."
	golangci-lint run ./...

# تولید مستندات
docs:
	@echo "📚 Generating documentation..."
	godoc -http=:6060 &
	open http://localhost:6060/pkg/github.com/lumix-ai/vts/

# ساخت Docker image
docker-build:
	@echo "🐳 Building Docker image..."
	docker build -t $(APP_NAME):$(VERSION) .
	docker tag $(APP_NAME):$(VERSION) $(APP_NAME):latest

# اجرای Docker container
docker-run:
	@echo "🐳 Running Docker container..."
	docker run -p 8080:8080 \
		-v ./data:/app/data \
		-e GOOGLE_API_KEY=${GOOGLE_API_KEY} \
		$(APP_NAME):latest

# استقرار روی Raspberry Pi
deploy-pi:
	@echo "🚀 Deploying to Raspberry Pi..."
	scp $(BUILD_DIR)/$(APP_NAME)-linux-armv5 pi@raspberrypi.local:~/lumix/
	scp -r $(DATA_DIR)/config pi@raspberrypi.local:~/lumix/data/
	ssh pi@raspberrypi.local "cd ~/lumix && ./$(APP_NAME)-linux-armv5 --offline"

# مانیتورینگ سیستم
monitor:
	@echo "📊 Monitoring system..."
	watch -n 1 "ps aux | grep lumix | grep -v grep && echo '---' && free -m && echo '---' && df -h /"

# ایجاد گزارش
report:
	@echo "📈 Generating report..."
	$(GO) tool cover -html=coverage.out -o coverage.html
	open coverage.html

# کمک
help:
	@echo "Lumix AI V-TS Build System"
	@echo ""
	@echo "Targets:"
	@echo "  setup        - Setup project directories and initial data"
	@echo "  build        - Build for all platforms"
	@echo "  build-linux  - Build for Linux"
	@echo "  build-arm    - Build for ARM (Raspberry Pi)"
	@echo "  build-windows- Build for Windows"
	@echo "  test         - Run unit tests"
	@echo "  test-integration - Run integration tests"
	@echo "  train        - Train initial model"
	@echo "  run          - Run in development mode"
	@echo "  run-offline  - Run in offline mode"
	@echo "  lint         - Run linter"
	@echo "  docker-build - Build Docker image"
	@echo "  docker-run   - Run Docker container"
	@echo "  deploy-pi    - Deploy to Raspberry Pi"
	@echo "  monitor      - Monitor system resources"
	@echo "  report       - Generate coverage report"
	@echo "  clean        - Clean build artifacts"
	@echo "  help         - Show this help"