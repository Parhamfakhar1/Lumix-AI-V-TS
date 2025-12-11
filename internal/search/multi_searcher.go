// internal/search/multi_searcher.go
package search

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"sort"
	"strings"
	"sync"
	"time"
	
	"github.com/lumix-ai/vts/internal/utils"
	"github.com/rs/zerolog/log"
	"golang.org/x/sync/semaphore"
)

// MultiSearcher - سیستم جستجوی ۹-کوئری موازی
type MultiSearcher struct {
	config         Config
	googleClient   *GoogleClient
	cache          *CacheManager
	queryAnalyzer  *QueryAnalyzer
	resultRanker   *ResultRanker
	semaphore      *semaphore.Weighted
	offlineMode    bool
	offlineDB      *OfflineKnowledgeBase
	stats          SearchStats
	mu             sync.RWMutex
}

type Config struct {
	GoogleAPIKey       string        `yaml:"google_api_key"`
	SearchEngineID     string        `yaml:"search_engine_id"`
	MaxResults         int           `yaml:"max_results"`
	QueryVariations    int           `yaml:"query_variations"`
	Timeout            time.Duration `yaml:"timeout"`
	RetryAttempts      int           `yaml:"retry_attempts"`
	RateLimitPerMinute int           `yaml:"rate_limit_per_minute"`
	CacheTTL           time.Duration `yaml:"cache_ttl"`
	MaxConcurrent      int           `yaml:"max_concurrent"`
}

type SearchResult struct {
	ID         string    `json:"id"`
	Title      string    `json:"title"`
	Snippet    string    `json:"snippet"`
	Link       string    `json:"link"`
	Source     string    `json:"source"`
	Relevance  float64   `json:"relevance"`
	Confidence float64   `json:"confidence"`
	Language   string    `json:"language"`
	Timestamp  time.Time `json:"timestamp"`
	Entities   []Entity  `json:"entities"`
	Summary    string    `json:"summary"`
	Categories []string  `json:"categories"`
}

type Entity struct {
	Text     string `json:"text"`
	Type     string `json:"type"`
	Score    float64 `json:"score"`
}

func NewMultiSearcher(config Config) *MultiSearcher {
	return &MultiSearcher{
		config:        config,
		googleClient:  NewGoogleClient(config.GoogleAPIKey, config.SearchEngineID),
		cache:         NewCacheManager(config.CacheTTL),
		queryAnalyzer: NewQueryAnalyzer(),
		resultRanker:  NewResultRanker(),
		semaphore:     semaphore.NewWeighted(int64(config.MaxConcurrent)),
		offlineDB:     NewOfflineKnowledgeBase(),
		stats:         SearchStats{},
	}
}

func (ms *MultiSearcher) Search(ctx context.Context, query string, options SearchOptions) ([]SearchResult, error) {
	ms.mu.Lock()
	ms.stats.TotalQueries++
	ms.mu.Unlock()
	
	startTime := time.Now()
	
	// بررسی کش
	cacheKey := ms.generateCacheKey(query, options)
	if cached, found := ms.cache.Get(cacheKey); found && !options.ForceRefresh {
		log.Debug().Str("query", query).Msg("Cache hit")
		ms.updateStats(true, time.Since(startTime))
		return cached, nil
	}
	
	// بررسی حالت آفلاین
	if ms.offlineMode || !utils.IsOnline() {
		log.Info().Str("query", query).Msg("Offline mode activated")
		return ms.searchOffline(query, options)
	}
	
	// تولید ۹ کوئری مختلف
	queries := ms.generate9Queries(query, options)
	
	// اجرای جستجوی موازی
	results := ms.executeParallelSearch(ctx, queries, options)
	
	// ادغام و رتبه‌بندی نتایج
	mergedResults := ms.mergeAndRankResults(results, query)
	
	// ذخیره در کش
	ms.cache.Set(cacheKey, mergedResults)
	
	// ذخیره در دانش آفلاین
	if options.SaveToKnowledgeBase {
		go ms.saveToKnowledgeBase(query, mergedResults)
	}
	
	ms.updateStats(false, time.Since(startTime))
	
	log.Info().
		Str("query", query).
		Int("total_results", len(mergedResults)).
		Dur("duration", time.Since(startTime)).
		Msg("Search completed")
	
	return mergedResults, nil
}

func (ms *MultiSearcher) generate9Queries(originalQuery string, options SearchOptions) []string {
	var queries []string
	
	// تحلیل کوئری اصلی
	analysis := ms.queryAnalyzer.Analyze(originalQuery)
	
	// 3 دسته‌بندی × 3 سطح جزئیات = 9 کوئری
	
	// دسته 1: کوئری‌های مستقیم
	queries = append(queries,
		originalQuery, // سطح 1: اصلی
		ms.expandQuery(originalQuery, 1), // سطح 2: گسترش یافته
		ms.specializeQuery(originalQuery, analysis), // سطح 3: تخصصی
	)
	
	// دسته 2: کوئری‌های مفهومی
	conceptual := ms.conceptualizeQuery(originalQuery, analysis)
	queries = append(queries,
		conceptual,
		ms.addContext(conceptual, "تعریف"),
		ms.addContext(conceptual, "آموزش"),
	)
	
	// دسته 3: کوئری‌های عملیاتی
	operational := ms.operationalizeQuery(originalQuery, analysis)
	queries = append(queries,
		operational,
		ms.addContext(operational, "راهنمایی"),
		ms.addContext(operational, "تجربه"),
	)
	
	// محدود کردن به 9 کوئری
	if len(queries) > 9 {
		queries = queries[:9]
	}
	
	// فیلتر کردن کوئری‌های تکراری
	return ms.deduplicateQueries(queries)
}

func (ms *MultiSearcher) executeParallelSearch(ctx context.Context, queries []string, options SearchOptions) [][]SearchResult {
	var wg sync.WaitGroup
	results := make([][]SearchResult, len(queries))
	errors := make([]error, len(queries))
	
	for i, query := range queries {
		wg.Add(1)
		
		go func(idx int, q string) {
			defer wg.Done()
			
			// محدود کردن همزمانی
			if err := ms.semaphore.Acquire(ctx, 1); err != nil {
				errors[idx] = err
				return
			}
			defer ms.semaphore.Release(1)
			
			// اجرای جستجو با قابلیت تکرار
			var res []SearchResult
			var err error
			
			for attempt := 0; attempt < ms.config.RetryAttempts; attempt++ {
				res, err = ms.googleClient.Search(ctx, q, options)
				if err == nil {
					break
				}
				
				log.Warn().
					Str("query", q).
					Int("attempt", attempt+1).
					Err(err).
					Msg("Search attempt failed")
				
				if attempt < ms.config.RetryAttempts-1 {
					time.Sleep(time.Duration(attempt+1) * time.Second)
				}
			}
			
			if err != nil {
				errors[idx] = err
				return
			}
			
			// پردازش نتایج
			processed := ms.processResults(res, q)
			results[idx] = processed
			
		}(i, query)
	}
	
	wg.Wait()
	
	// بررسی خطاها
	for i, err := range errors {
		if err != nil {
			log.Error().
				Str("query", queries[i]).
				Err(err).
				Msg("Search failed")
		}
	}
	
	return results
}

func (ms *MultiSearcher) processResults(rawResults []GoogleResult, query string) []SearchResult {
	var processed []SearchResult
	
	for _, result := range rawResults {
		// استخراج موجودیت‌ها
		entities := ms.extractEntities(result.Snippet, result.Title)
		
		// تولید خلاصه
		summary := ms.generateSummary(result.Snippet, query)
		
		// تشخیص زبان
		language := ms.detectLanguage(result.Snippet)
		
		// محاسبه ارتباط
		relevance := ms.calculateRelevance(result, query)
		
		processed = append(processed, SearchResult{
			ID:         utils.GenerateID(),
			Title:      ms.cleanText(result.Title),
			Snippet:    ms.cleanText(result.Snippet),
			Link:       result.Link,
			Source:     "google",
			Relevance:  relevance,
			Confidence: ms.calculateConfidence(result),
			Language:   language,
			Timestamp:  time.Now(),
			Entities:   entities,
			Summary:    summary,
			Categories: ms.categorizeResult(result, query),
		})
	}
	
	return processed
}

func (ms *MultiSearcher) mergeAndRankResults(allResults [][]SearchResult, originalQuery string) []SearchResult {
	// ادغام تمام نتایج
	var merged []SearchResult
	seenLinks := make(map[string]bool)
	
	for _, results := range allResults {
		for _, result := range results {
			if !seenLinks[result.Link] {
				seenLinks[result.Link] = true
				merged = append(merged, result)
			} else {
				// افزایش امتیاز نتایج تکراری
				for i := range merged {
					if merged[i].Link == result.Link {
						merged[i].Relevance *= 1.2
						break
					}
				}
			}
		}
	}
	
	// رتبه‌بندی نتایج
	ms.resultRanker.Rank(merged, originalQuery)
	
	// مرتب‌سازی بر اساس امتیاز نهایی
	sort.Slice(merged, func(i, j int) bool {
		return merged[i].Relevance > merged[j].Relevance
	})
	
	// محدود کردن تعداد نتایج
	if len(merged) > ms.config.MaxResults {
		merged = merged[:ms.config.MaxResults]
	}
	
	return merged
}

func (ms *MultiSearcher) searchOffline(query string, options SearchOptions) ([]SearchResult, error) {
	// جستجو در دانش آفلاین
	results, err := ms.offlineDB.Search(query, options)
	if err != nil {
		return nil, err
	}
	
	// اگر نتیجه‌ای یافت نشد، از مدل زبانی استفاده کن
	if len(results) == 0 {
		generated := ms.generateFromLanguageModel(query)
		if generated != "" {
			results = append(results, SearchResult{
				ID:        utils.GenerateID(),
				Title:     "پاسخ بر اساس دانش داخلی",
				Snippet:   generated,
				Source:    "offline_model",
				Relevance: 0.7,
				Timestamp: time.Now(),
			})
		}
	}
	
	return results, nil
}

func (ms *MultiSearcher) saveToKnowledgeBase(query string, results []SearchResult) {
	for _, result := range results {
		knowledge := KnowledgeEntry{
			Query:      query,
			Result:     result,
			AccessedAt: time.Now(),
			AccessCount: 1,
		}
		
		if err := ms.offlineDB.Store(knowledge); err != nil {
			log.Error().Err(err).Msg("Failed to save to knowledge base")
		}
	}
}

// توابع کمکی
func (ms *MultiSearcher) generateCacheKey(query string, options SearchOptions) string {
	key := fmt.Sprintf("%s:%v:%v:%v",
		query,
		options.Language,
		options.Freshness,
		options.MaxResults,
	)
	return utils.HashSHA256(key)
}

func (ms *MultiSearcher) updateStats(cacheHit bool, duration time.Duration) {
	ms.mu.Lock()
	defer ms.mu.Unlock()
	
	if cacheHit {
		ms.stats.CacheHits++
	}
	ms.stats.TotalDuration += duration
	ms.stats.AverageDuration = ms.stats.TotalDuration / time.Duration(ms.stats.TotalQueries)
}

func (ms *MultiSearcher) cleanText(text string) string {
	// حذف HTML tags
	text = utils.StripHTML(text)
	
	// نرمال‌سازی فاصله‌ها
	text = utils.NormalizeSpaces(text)
	
	// حذف کاراکترهای غیرمجاز
	text = utils.RemoveInvalidChars(text)
	
	return strings.TrimSpace(text)
}