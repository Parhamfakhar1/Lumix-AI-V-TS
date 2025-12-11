// internal/search/intelligent_searcher.go
package search

import (
	"context"
	"strings"
	"sync/atomic"
	"time"
	
	"github.com/lumix-ai/vts/internal/core"
	"github.com/lumix-ai/vts/internal/learning"
	"github.com/lumix-ai/vts/internal/memory"
)

// IntelligentSearcher - جستجوگر ۳-لایه با یادگیری تطبیقی
type IntelligentSearcher struct {
	config        SearchConfig
	googleClient  *GoogleClient
	cache         *AdaptiveCache
	queryLearner  *QueryLearningEngine
	resultAnalyzer *ResultAnalyzer
	knowledgeBase *memory.NeuralMemory
	userProfiles  *UserProfileManager
	
	// آمار پیشرفته
	stats        *SearchStatistics
	failedSearches *FailedSearchTracker
	successPatterns *SuccessPatternLearner
}

// AdaptiveCache - کش تطبیقی با یادگیری الگوها
type AdaptiveCache struct {
	mainCache    map[string]*CachedResult
	patternCache map[string]*SearchPattern
	temporalCache *TemporalCache
	adaptiveTTL  map[string]time.Duration // TTL پویا بر اساس محبوبیت
	
	hitStats      map[string]int
	missStats     map[string]int
	relevanceStats map[string]float32
}

func NewIntelligentSearcher(config SearchConfig, knowledgeBase *memory.NeuralMemory) *IntelligentSearcher {
	return &IntelligentSearcher{
		config:        config,
		googleClient:  NewGoogleClient(config),
		cache: &AdaptiveCache{
			mainCache:     make(map[string]*CachedResult),
			patternCache:  make(map[string]*SearchPattern),
			temporalCache: NewTemporalCache(24*time.Hour),
			adaptiveTTL:   make(map[string]time.Duration),
			hitStats:      make(map[string]int),
			missStats:     make(map[string]int),
			relevanceStats: make(map[string]float32),
		},
		queryLearner:   NewQueryLearningEngine(knowledgeBase),
		resultAnalyzer: NewResultAnalyzer(knowledgeBase),
		knowledgeBase:  knowledgeBase,
		userProfiles:   NewUserProfileManager(),
		stats:          NewSearchStatistics(),
		failedSearches: NewFailedSearchTracker(),
		successPatterns: NewSuccessPatternLearner(),
	}
}

// SearchWithLearning - جستجو با یادگیری تطبیقی
func (is *IntelligentSearcher) SearchWithLearning(ctx context.Context, 
	query string, userID string, sessionContext *SessionContext) (*SearchResponse, error) {
	
	startTime := time.Now()
	
	// 1. تحلیل کوئری با استفاده از دانش موجود
	queryAnalysis := is.analyzeQuery(query, userID)
	
	// 2. تولید کوئری‌های بهینه‌شده (لایه‌بندی)
	optimizedQueries := is.generateOptimizedQueries(queryAnalysis, 3) // 3 لایه
	
	// 3. اجرای جستجوی لایه‌ای
	var allResults []*EnrichedResult
	for layer, queries := range optimizedQueries {
		layerResults, err := is.searchLayer(ctx, queries, layer, sessionContext)
		if err != nil {
			is.failedSearches.RecordFailure(query, layer, err)
			continue
		}
		
		// 4. غنی‌سازی نتایج با دانش داخلی
		enrichedResults := is.enrichResults(layerResults, queryAnalysis)
		allResults = append(allResults, enrichedResults...)
		
		// 5. اگر نتایج لایه کافی بود، ادامه نده
		if len(enrichedResults) >= is.config.MinResultsPerLayer && layer < 2 {
			break
		}
	}
	
	// 6. ادغام و رتبه‌بندی هوشمند
	mergedResults := is.mergeAndRankResults(allResults, queryAnalysis)
	
	// 7. یادگیری از این جستجو
	is.learnFromSearch(query, mergedResults, queryAnalysis, userID)
	
	// 8. به‌روزرسانی پروفایل کاربر
	is.updateUserProfile(userID, query, mergedResults)
	
	duration := time.Since(startTime)
	is.stats.RecordSearch(query, len(mergedResults), duration, queryAnalysis.Confidence)
	
	return &SearchResponse{
		Query:         query,
		Results:       mergedResults,
		QueryAnalysis: queryAnalysis,
		SearchTime:    duration,
		TotalLayers:   len(optimizedQueries),
		UsedCache:     is.cache.GetHitRate(query),
		Confidence:    is.calculateConfidence(mergedResults),
	}, nil
}

// generateOptimizedQueries - تولید ۳ لایه کوئری بهینه
func (is *IntelligentSearcher) generateOptimizedQueries(analysis *QueryAnalysis, layers int) map[int][]string {
	queriesByLayer := make(map[int][]string)
	
	// لایه ۱: کوئری‌های مستقیم و عمومی
	queriesByLayer[1] = []string{
		analysis.OriginalQuery,
		is.expandQuery(analysis.OriginalQuery, analysis.Context),
		is.simplifyQuery(analysis.OriginalQuery),
	}
	
	// لایه ۲: کوئری‌های تخصصی‌شده
	if len(analysis.Keywords) > 0 {
		queriesByLayer[2] = []string{
			is.createExpertQuery(analysis.Keywords, analysis.Domain),
			is.createComparativeQuery(analysis.Keywords),
			is.createHowToQuery(analysis.OriginalQuery),
		}
	}
	
	// لایه ۳: کوئری‌های استنتاجی از دانش موجود
	if len(analysis.RelatedConcepts) > 0 {
		inferredQueries := is.inferQueriesFromKnowledge(analysis.RelatedConcepts, 3)
		queriesByLayer[3] = inferredQueries
	}
	
	return queriesByLayer
}

// enrichResults - غنی‌سازی نتایج با دانش داخلی
func (is *IntelligentSearcher) enrichResults(results []*SearchResult, 
	analysis *QueryAnalysis) []*EnrichedResult {
	
	var enriched []*EnrichedResult
	
	for _, result := range results {
		enrichedResult := &EnrichedResult{
			BaseResult:    result,
			Relevance:     is.calculateRelevance(result, analysis),
			TrustScore:    is.calculateTrustScore(result.Source),
			Freshness:     is.calculateFreshness(result.Timestamp),
			RelatedConcepts: is.extractConcepts(result.Content),
			Summary:       is.generateIntelligentSummary(result, analysis),
			KeyPoints:     is.extractKeyPoints(result.Content, analysis.Keywords),
			Contradictions: is.checkContradictions(result, analysis),
			Gaps:          is.identifyKnowledgeGaps(result, analysis),
		}
		
		// افزودن استنتاج‌های مبتنی بر دانش
		if inferences := is.knowledgeBase.Infer(enrichedResult.RelatedConcepts, 2); len(inferences) > 0 {
			enrichedResult.Inferences = inferences
		}
		
		enriched = append(enriched, enrichedResult)
	}
	
	return enriched
}

// learnFromSearch - یادگیری از جستجوی انجام شده
func (is *IntelligentSearcher) learnFromSearch(query string, results []*RankedResult, 
	analysis *QueryAnalysis, userID string) {
	
	// 1. یادگیری الگوهای کوئری موفق
	if len(results) > 0 && results[0].Relevance > 0.7 {
		is.successPatterns.LearnPattern(query, analysis, results)
		
		// تقویت ارتباطات در دانش پایه
		for _, result := range results {
			if result.Relevance > 0.8 {
				for _, concept := range result.RelatedConcepts {
					is.knowledgeBase.LearnAssociation(
						query, 
						concept, 
						"searched-for", 
						result.Relevance,
					)
				}
			}
		}
	}
	
	// 2. به‌روزرسانی کش تطبیقی
	is.cache.UpdateAdaptiveTTL(query, len(results), avgRelevance(results))
	
	// 3. یادگیری ترجیحات کاربر
	if userID != "" {
		preferredSources := is.extractPreferredSources(results, userID)
		is.userProfiles.UpdatePreferences(userID, "preferred_sources", preferredSources)
	}
}

// mergeAndRankResults - ادغام و رتبه‌بندی هوشمند نتایج
func (is *IntelligentSearcher) mergeAndRankResults(results []*EnrichedResult, 
	analysis *QueryAnalysis) []*RankedResult {
	
	// گروه‌بندی نتایج بر اساس منبع
	groupedResults := make(map[string][]*EnrichedResult)
	for _, result := range results {
		groupedResults[result.BaseResult.Source] = append(
			groupedResults[result.BaseResult.Source], 
			result,
		)
	}
	
	var rankedResults []*RankedResult
	
	// رتبه‌بندی در هر گروه
	for source, sourceResults := range groupedResults {
		// نرمال‌سازی امتیازها در هر گروه
		normalized := is.normalizeScores(sourceResults)
		
		// اعمال وزن منبع
		sourceWeight := is.getSourceWeight(source)
		for i := range normalized {
			normalized[i].CompositeScore *= sourceWeight
		}
		
		// انتخاب بهترین نتایج از هر گروه
		bestFromSource := is.selectBestResults(normalized, 2)
		rankedResults = append(rankedResults, bestFromSource...)
	}
	
	// مرتب‌سازی نهایی بر اساس امتیاز ترکیبی
	sort.Slice(rankedResults, func(i, j int) bool {
		return rankedResults[i].CompositeScore > rankedResults[j].CompositeScore
	})
	
	// حذف تکراری‌ها و ترکیب نتایج مشابه
	rankedResults = is.deduplicateAndMerge(rankedResults)
	
	// محدود کردن تعداد نتایج نهایی
	if len(rankedResults) > is.config.MaxTotalResults {
		rankedResults = rankedResults[:is.config.MaxTotalResults]
	}
	
	return rankedResults
}