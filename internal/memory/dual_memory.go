// internal/memory/dual_memory.go
package memory

type DualMemory struct {
    // حافظه سریع (SQLite)
    FastMemory *sql.DB // برای دسترسی سریع
    
    // حافظه آرشیو (فایل‌های append-only)
    ArchiveDir string // data/archive/
    
    // کش در RAM (محدود)
    Cache      *lru.Cache // حداکثر 1000 آیتم
}

func (dm *DualMemory) Store(conversation *Conversation) error {
    // 1. ذخیره در SQLite برای دسترسی سریع
    dm.storeFast(conversation)
    
    // 2. اضافه به آرشیو روزانه
    dm.appendToArchive(conversation)
    
    // 3. اگر آرشیو بزرگ شد، فشرده‌سازی
    if dm.archiveSize() > 1_000_000_000 { // 1GB
        dm.compressOldArchives()
    }
}