// internal/learning/adaptive_learner.go
package learning

import (
	"math"
	"time"
	
	"github.com/lumix-ai/vts/internal/core"
	"github.com/lumix-ai/vts/internal/memory"
)

// AdaptiveLearner - سیستم یادگیری تطبیقی با چندین استراتژی
type AdaptiveLearner struct {
	strategies        map[string]LearningStrategy
	activeStrategies  []string
	strategyWeights   map[string]float32
	metaLearner       *MetaLearningController
	knowledgeBase     *memory.NeuralMemory
	performanceTracker *PerformanceTracker
	curriculumManager *CurriculumManager
	
	// پارامترهای یادگیری پویا
	learningRate      float32
	momentum          float32
	explorationRate   float32
	forgettingRate    float32
	consolidationRate float32
}

type LearningStrategy interface {
	Name() string
	Learn(sample *LearningSample, context *LearningContext) *LearningResult
	CanApply(sample *LearningSample) bool
	Confidence() float32
	UpdateWeight(delta float32)
}

// استراتژی‌های مختلف یادگیری
var (
	// یادگیری از طریق تقلید
	ImitationLearning = &ImitationStrategy{
		baseWeight: 0.3,
		minSamples: 10,
	}
	
	// یادگیری از طریق کشف
	ExploratoryLearning = &ExploratoryStrategy{
		baseWeight: 0.2,
		explorationBonus: 0.1,
	}
	
	// یادگیری از طریق تمرین
	PracticeLearning = &PracticeStrategy{
		baseWeight: 0.25,
		spacingFactor: 1.5,
	}
	
	// یادگیری از طریق توضیح
	ExplanationLearning = &ExplanationStrategy{
		baseWeight: 0.15,
		clarityThreshold: 0.7,
	}
	
	// یادگیری از طریق خطا
	ErrorDrivenLearning = &ErrorDrivenStrategy{
		baseWeight: 0.1,
		errorSensitivity: 0.8,
	}
)

func NewAdaptiveLearner(knowledgeBase *memory.NeuralMemory) *AdaptiveLearner {
	al := &AdaptiveLearner{
		strategies: map[string]LearningStrategy{
			"imitation":   ImitationLearning,
			"exploratory": ExploratoryLearning,
			"practice":    PracticeLearning,
			"explanation": ExplanationLearning,
			"error":       ErrorDrivenLearning,
		},
		knowledgeBase:     knowledgeBase,
		performanceTracker: NewPerformanceTracker(),
		curriculumManager: NewCurriculumManager(),
		metaLearner:       NewMetaLearningController(),
		
		// مقداردهی اولیه پارامترها
		learningRate:      0.01,
		momentum:          0.9,
		explorationRate:   0.1,
		forgettingRate:    0.001,
		consolidationRate: 0.05,
	}
	
	// بارگذاری وزن استراتژی‌ها از حافظه
	al.loadStrategyWeights()
	
	return al
}

// LearnAdaptively - یادگیری تطبیقی با انتخاب خودکار استراتژی
func (al *AdaptiveLearner) LearnAdaptively(sample *LearningSample, 
	context *LearningContext) *LearningResult {
	
	// 1. ارزیابی نمونه برای تعیین بهترین استراتژی
	applicableStrategies := al.evaluateStrategies(sample)
	
	// 2. انتخاب استراتژی‌ها بر اساس وزن و اعتبار
	selectedStrategies := al.selectStrategies(applicableStrategies, 2) // 2 استراتژی برتر
	
	// 3. اجرای یادگیری با استراتژی‌های انتخاب شده
	var results []*LearningResult
	for _, strategy := range selectedStrategies {
		result := strategy.Learn(sample, context)
		if result.Success {
			results = append(results, result)
			
			// 4. تقویت استراتژی موفق
			strategy.UpdateWeight(0.05)
			
			// 5. تثبیت دانش کسب شده
			al.consolidateLearning(result, strategy)
		} else {
			// تضعیف استراتژی ناموفق
			strategy.UpdateWeight(-0.03)
		}
	}
	
	// 6. ترکیب نتایج از استراتژی‌های مختلف
	combinedResult := al.combineResults(results)
	
	// 7. به‌روزرسانی پارامترهای یادگیری بر اساس عملکرد
	al.updateLearningParameters(combinedResult)
	
	// 8. ذخیره تجربه یادگیری برای فرا-یادگیری
	al.metaLearner.RecordExperience(sample, selectedStrategies, combinedResult)
	
	return combinedResult
}

// evaluateStrategies - ارزیابی استراتژی‌های applicable
func (al *AdaptiveLearner) evaluateStrategies(sample *LearningSample) []StrategyEvaluation {
	var evaluations []StrategyEvaluation
	
	for name, strategy := range al.strategies {
		if strategy.CanApply(sample) {
			score := al.calculateStrategyScore(strategy, sample)
			evaluations = append(evaluations, StrategyEvaluation{
				Strategy: strategy,
				Name:     name,
				Score:    score,
				Weight:   al.strategyWeights[name],
			})
		}
	}
	
	// مرتب‌سازی بر اساس امتیاز
	sort.Slice(evaluations, func(i, j int) bool {
		// ترکیب امتیاز و وزن
		combinedI := evaluations[i].Score * evaluations[i].Weight
		combinedJ := evaluations[j].Score * evaluations[j].Weight
		return combinedI > combinedJ
	})
	
	return evaluations
}

// MetaLearningController - کنترل‌گر فرا-یادگیری
type MetaLearningController struct {
	experiences     []*MetaExperience
	patternDetector *PatternDetector
	ruleExtractor   *RuleExtractor
	advisor         *LearningAdvisor
}

type MetaExperience struct {
	Timestamp     time.Time
	Sample        *LearningSample
	Strategies    []LearningStrategy
	Result        *LearningResult
	Context       *LearningContext
	Performance   float32
	Lessons       []string
}

func (mlc *MetaLearningController) RecordExperience(sample *LearningSample, 
	strategies []LearningStrategy, result *LearningResult) {
	
	experience := &MetaExperience{
		Timestamp:  time.Now(),
		Sample:     sample,
		Strategies: strategies,
		Result:     result,
		Performance: al.calculatePerformance(result),
		Lessons:    mlc.extractLessons(sample, strategies, result),
	}
	
	mlc.experiences = append(mlc.experiences, experience)
	
	// تشخیص الگوهای یادگیری موفق
	if len(mlc.experiences) >= 100 {
		mlc.analyzePatterns()
	}
}

func (mlc *MetaLearningController) analyzePatterns() {
	// گروه‌بندی تجربیات بر اساس نوع
	grouped := make(map[string][]*MetaExperience)
	for _, exp := range mlc.experiences {
		key := exp.Sample.Type + "_" + exp.Context.Domain
		grouped[key] = append(grouped[key], exp)
	}
	
	// استخراج قوانین فرا-یادگیری
	for key, experiences := range grouped {
		if len(experiences) >= 10 {
			rules := mlc.ruleExtractor.ExtractRules(experiences)
			mlc.advisor.AddRules(key, rules)
		}
	}
	
	// پیشنهاد بهینه‌سازی‌ها
	optimizations := mlc.advisor.GenerateOptimizations()
	al.applyOptimizations(optimizations)
}

// CurriculumManager - مدیریت برنامه درسی پویا
type CurriculumManager struct {
	knowledgeMap    *KnowledgeGraph
	difficultyCurve *DifficultyCurve
	prerequisiteMap map[string][]string
	progressTracker *ProgressTracker
	adaptiveScheduler *AdaptiveScheduler
	
	currentFocus    []string
	learningPath    []*LearningUnit
	masteryLevels   map[string]float32
	gaps            []*KnowledgeGap
}

func (cm *CurriculumManager) PlanLearningPath(currentKnowledge map[string]float32, 
	goals []string, constraints *LearningConstraints) []*LearningUnit {
	
	// 1. شناسایی شکاف‌های دانش
	cm.identifyGaps(currentKnowledge, goals)
	
	// 2. تعیین اولویت‌های یادگیری
	priorities := cm.prioritizeGaps()
	
	// 3. ساخت مسیر یادگیری
	var path []*LearningUnit
	for _, gap := range priorities {
		units := cm.createLearningUnits(gap, constraints)
		path = append(path, units...)
	}
	
	// 4. بهینه‌سازی توالی با در نظر گرفتن پیش‌نیازها
	optimizedPath := cm.optimizeSequence(path)
	
	// 5. افزودن مرور و تثبیت
	cm.addReviewAndConsolidation(optimizedPath)
	
	return optimizedPath
}

func (cm *CurriculumManager) createLearningUnits(gap *KnowledgeGap, 
	constraints *LearningConstraints) []*LearningUnit {
	
	var units []*LearningUnit
	
	// تعیین سطح دشواری مناسب
	difficulty := cm.difficultyCurve.CalculateOptimalDifficulty(
		cm.masteryLevels[gap.ParentConcept],
		gap.Complexity,
	)
	
	// ایجاد واحدهای یادگیری تدریجی
	concepts := cm.decomposeConcept(gap.Concept, difficulty.Levels)
	
	for i, concept := range concepts {
		unit := &LearningUnit{
			ID:           fmt.Sprintf("%s_%d", gap.ID, i),
			Concept:      concept,
			Difficulty:   difficulty.Base + (float32(i) * difficulty.Step),
			Duration:     cm.calculateDuration(concept, constraints),
			Prerequisites: cm.prerequisiteMap[concept],
			LearningObjectives: cm.defineObjectives(concept),
			Assessment:    cm.createAssessment(concept),
			Resources:     cm.selectResources(concept, constraints),
			Adaptive:      true,
		}
		
		units = append(units, unit)
	}
	
	return units
}

// سیستم تثبیت و مرور فضایی‌-زمانی
type SpacedRepetitionSystem struct {
	memoryModels   map[string]*ForgettingCurve
	intervals      map[string][]time.Duration
	successRates   map[string]float32
	adaptationRate float32
	
	reviewQueue    *PriorityQueue
	scheduler      *ReviewScheduler
}

func (srs *SpacedRepetitionSystem) ScheduleReview(item *MemoryItem, 
	performance float32) time.Time {
	
	// محاسبه فاصله مرور بعدی بر اساس منحنی فراموشی
	curve := srs.getForgettingCurve(item.Type)
	interval := curve.CalculateNextInterval(
		item.Retention,
		performance,
		item.ReviewCount,
	)
	
	// تطبیق فاصله بر اساس نرخ موفقیت
	adjustedInterval := srs.adaptInterval(interval, item.ID, performance)
	
	// برنامه‌ریزی مرور بعدی
	nextReview := time.Now().Add(adjustedInterval)
	srs.reviewQueue.Push(item.ID, nextReview, item.Priority)
	
	return nextReview
}

func (srs *SpacedRepetitionSystem) adaptInterval(baseInterval time.Duration, 
	itemID string, performance float32) time.Duration {
	
	// بررسی تاریخچه عملکرد این آیتم
	history := srs.successRates[itemID]
	if len(history) >= 3 {
		avgSuccess := average(history)
		
		// اگر عملکرد خوب است، افزایش فاصله
		if avgSuccess > 0.8 && performance > 0.7 {
			return baseInterval * 2
		}
		
		// اگر عملکرد ضعیف است، کاهش فاصله
		if avgSuccess < 0.5 || performance < 0.4 {
			return baseInterval / 2
		}
	}
	
	return baseInterval
}