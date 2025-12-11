# Dockerfile
FROM golang:1.21-alpine AS builder

# نصب وابستگی‌های build
RUN apk add --no-cache \
    git \
    make \
    gcc \
    musl-dev

WORKDIR /app

# کپی فایل‌های پروژه
COPY go.mod go.sum ./
RUN go mod download

COPY . .

# کامپایل
RUN make build-linux

# مرحله نهایی
FROM alpine:3.18

# نصب runtime dependencies
RUN apk add --no-cache \
    ca-certificates \
    tzdata \
    sqlite \
    && update-ca-certificates

WORKDIR /app

# کپی باینری از مرحله builder
COPY --from=builder /app/bin/lumix-ai-vts-linux-amd64 /app/lumix
COPY --from=builder /app/data/config/default.yaml /app/config/default.yaml

# ایجاد volume برای داده‌ها
VOLUME ["/app/data"]

# ایجاد کاربر non-root
RUN addgroup -g 1000 lumix && \
    adduser -u 1000 -G lumix -D lumix && \
    chown -R lumix:lumix /app

USER lumix

# پورت اکسپوز
EXPOSE 8080

# متغیرهای محیطی
ENV LUMIX_MODE=production \
    LUMIX_PORT=8080 \
    LUMIX_DATA_DIR=/app/data \
    LUMIX_CONFIG=/app/config/default.yaml

# اجرای برنامه
ENTRYPOINT ["/app/lumix"]
CMD ["--config", "/app/config/default.yaml"]