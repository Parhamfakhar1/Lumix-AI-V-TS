// internal/learning/incremental.go
package learning

type IncrementalLearner struct {
    Model        *model.NanoTransformer
    Memory       *memory.DualMemory
    LearningRate float32
}

func (il *IncrementalLearner) LearnFromConversation(conv *Conversation) {
    // 1. استخراج الگوهای جدید
    patterns := il.extractPatterns(conv)
    
    // 2. اضافه کردن به حافظه کوتاه‌مدت
    il.Memory.StoreShortTerm(patterns)
    
    // 3. اگر به 100 نمونه رسید، آموزش سریع
    if il.Memory.ShortTermCount() >= 100 {
        il.quickTrain(il.Memory.GetRecent(100))
    }
    
    // 4. اگر به 1000 نمونه رسید، آموزش عمیق‌تر
    if il.Memory.TotalCount()%1000 == 0 {
        go il.deepTrain() // در background اجرا شود
    }
}

func (il *IncrementalLearner) quickTrain(samples []TrainingExample) {
    // آموزش سریع 10 دقیقه‌ای
    start := time.Now()
    for time.Since(start) < 10*time.Minute {
        il.Model.TrainBatch(samples, il.LearningRate)
    }
}