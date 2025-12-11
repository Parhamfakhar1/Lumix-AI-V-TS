# راهنمای راه‌اندازی Lumix AI V-TS

## حداقل سیستم مورد نیاز:
- CPU: 2 هسته @ 800MHz
- RAM: 100MB
- فضای دیسک: 500MB
- اینترنت (اختیاری)

## نصب:
1. دانلود باینری مناسب از releases
2. خروجی گرفتن: `tar -xzf lumix-v1.0.0.tar.gz`
3. اجرا: `./lumix --data-dir=./data`

## آموزش اولیه:
# مدل از قبل روی 10,000 داده آموزش دیده است
# برای آموزش بیشتر:
./lumix --train --epochs=5

## حالت آفلاین:
./lumix --offline --knowledge-file=base_knowledge.gob