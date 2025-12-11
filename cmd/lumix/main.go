// cmd/lumix/main.go
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"syscall"
	"time"
	
	"github.com/lumix-ai/vts/internal/core"
	"github.com/lumix-ai/vts/internal/learning"
	"github.com/lumix-ai/vts/internal/memory"
	"github.com/lumix-ai/vts/internal/model"
	"github.com/lumix-ai/vts/internal/search"
	"github.com/lumix-ai/vts/internal/utils"
	"github.com/lumix-ai/vts/pkg/api"
	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
	"gopkg.in/yaml.v3"
)

type Config struct {
	System      SystemConfig      `yaml:"system"`
	Model       model.Config      `yaml:"model"`
	Search      search.Config     `yaml:"search"`
	Memory      memory.Config     `yaml:"memory"`
	Learning    learning.Config   `yaml:"learning"`
	Performance PerformanceConfig `yaml:"performance"`
	Offline     OfflineConfig     `yaml:"offline"`
	Logging     LoggingConfig     `yaml:"logging"`
	API         api.Config        `yaml:"api"`
}

type SystemConfig struct {
	Name    string `yaml:"name"`
	Version string `yaml:"version"`
	Mode    string `yaml:"mode"`
	Debug   bool   `yaml:"debug"`
}

type PerformanceConfig struct {
	MaxGoroutines     int  `yaml:"max_goroutines"`
	MemoryLimitMB     int  `yaml:"memory_limit_mb"`
	CPUCores          int  `yaml:"cpu_cores"`
	GPUEnabled        bool `yaml:"gpu_enabled"`
	Quantization      bool `yaml:"quantization_enabled"`
	Pruning           bool `yaml:"pruning_enabled"`
}

type OfflineConfig struct {
	Enabled           bool   `yaml:"enabled"`
	KnowledgeBasePath string `yaml:"knowledge_base_path"`
	FallbackMode      string `yaml:"fallback_mode"`
	SyncOnReconnect   bool   `yaml:"sync_on_reconnect"`
}

type LoggingConfig struct {
	Level      string `yaml:"level"`
	Format     string `yaml:"format"`
	OutputPath string `yaml:"output_path"`
	MaxSizeMB  int    `yaml:"max_size_mb"`
	MaxAgeDays int    `yaml:"max_age_days"`
	Compression bool  `yaml:"compression"`
}

var (
	configFile  = flag.String("config", "config/default.yaml", "Configuration file path")
	modelPath   = flag.String("model", "data/models/pretrained_10k.bin", "Pre-trained model path")
	dataPath    = flag.String("data", "data/training/", "Training data path")
	offlineMode = flag.Bool("offline", false, "Run in offline mode")
	port        = flag.Int("port", 8080, "API server port")
	verbose     = flag.Bool("verbose", false, "Enable verbose logging")
)

func main() {
	flag.Parse()
	
	// راه‌اندازی logger
	setupLogger()
	
	log.Info().Msg("🚀 Starting Lumix AI V-TS")
	log.Info().Msg("==============================")
	
	// بارگذاری تنظیمات
	config, err := loadConfig(*configFile)
	if err != nil {
		log.Fatal().Err(err).Msg("Failed to load configuration")
	}
	
	// تنظیم محدودیت‌های سیستم
	setSystemLimits(config)
	
	// ایجاد context با قابلیت cancel
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	
	// مدیریت سیگنال‌های سیستم
	setupSignalHandler(cancel)
	
	// نمایش اطلاعات سیستم
	printSystemInfo(config)
	
	// راه‌اندازی کامپوننت‌ها
	components, err := setupComponents(ctx, config)
	if err != nil {
		log.Fatal().Err(err).Msg("Failed to setup components")
	}
	
	// بارگذاری مدل آموزش‌دیده
	log.Info().Msg("Loading pre-trained model...")
	if err := components.Model.LoadCheckpoint(*modelPath); err != nil {
		log.Warn().Err(err).Msg("Failed to load pre-trained model, initializing new model")
		// آموزش اولیه با 10,000 داده
		if err := trainInitialModel(components.Model, *dataPath); err != nil {
			log.Fatal().Err(err).Msg("Failed to train initial model")
		}
	}
	
	// راه‌اندازی سرویس‌ها
	services, err := startServices(ctx, config, components)
	if err != nil {
		log.Fatal().Err(err).Msg("Failed to start services")
	}
	
	// راه‌اندازی API سرور
	apiServer, err := api.NewServer(config.API, components)
	if err != nil {
		log.Fatal().Err(err).Msg("Failed to create API server")
	}
	
	log.Info().Msgf("Starting API server on port %d", *port)
	go func() {
		if err := apiServer.Start(fmt.Sprintf(":%d", *port)); err != nil {
			log.Fatal().Err(err).Msg("API server failed")
		}
	}()
	
	// شروع یادگیری افزایشی در background
	if config.Learning.IncrementalEnabled {
		go startIncrementalLearning(ctx, components)
	}
	
	// شروع جمع‌آوری آمار
	go collectMetrics(ctx, components)
	
	log.Info().Msg("✅ Lumix AI V-TS is ready!")
	log.Info().Msg("==============================")
	
	// نگه داشتن برنامه فعال
	<-ctx.Done()
	
	// توقف تمیز
	shutdown(apiServer, services, components)
	
	log.Info().Msg("👋 Lumix AI V-TS shutdown complete")
}

func setupLogger() {
	zerolog.TimeFieldFormat = zerolog.TimeFormatUnix
	
	if *verbose {
		zerolog.SetGlobalLevel(zerolog.DebugLevel)
	} else {
		zerolog.SetGlobalLevel(zerolog.InfoLevel)
	}
	
	// استفاده از console writer برای توسعه
	output := zerolog.ConsoleWriter{
		Out:        os.Stderr,
		TimeFormat: time.RFC3339,
	}
	
	log.Logger = log.Output(output)
}

func loadConfig(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read config file: %w", err)
	}
	
	var config Config
	if err := yaml.Unmarshal(data, &config); err != nil {
		return nil, fmt.Errorf("failed to parse config: %w", err)
	}
	
	// اعتبارسنجی تنظیمات
	if err := validateConfig(&config); err != nil {
		return nil, err
	}
	
	return &config, nil
}

func validateConfig(config *Config) error {
	if config.Model.HiddenSize%config.Model.NumHeads != 0 {
		return fmt.Errorf("hidden_size must be divisible by num_heads")
	}
	
	if config.Performance.MemoryLimitMB < 100 {
		return fmt.Errorf("memory_limit_mb must be at least 100MB")
	}
	
	if config.Search.MaxResults > 50 {
		return fmt.Errorf("max_results cannot exceed 50")
	}
	
	return nil
}

func setSystemLimits(config *Config) {
	// تنظیم محدودیت حافظه
	if config.Performance.MemoryLimitMB > 0 {
		utils.SetMemoryLimit(config.Performance.MemoryLimitMB * 1024 * 1024)
	}
	
	// تنظیم محدودیت هسته‌های CPU
	if config.Performance.CPUCores > 0 {
		utils.SetCPUCores(config.Performance.CPUCores)
	}
	
	// تنظیم حداکثر goroutine
	if config.Performance.MaxGoroutines > 0 {
		utils.SetMaxGoroutines(config.Performance.MaxGoroutines)
	}
}

func setupSignalHandler(cancel context.CancelFunc) {
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
	
	go func() {
		sig := <-sigChan
		log.Info().Str("signal", sig.String()).Msg("Received shutdown signal")
		cancel()
		
		// اگر بعد از 5 ثانیه هنوز اجراست، force kill
		time.Sleep(5 * time.Second)
		log.Error().Msg("Force shutdown after timeout")
		os.Exit(1)
	}()
}

func printSystemInfo(config *Config) {
	log.Info().Msgf("System: %s v%s", config.System.Name, config.System.Version)
	log.Info().Msgf("Mode: %s", config.System.Mode)
	log.Info().Msgf("Model: %d layers, %d hidden, %d heads", 
		config.Model.NumLayers, config.Model.HiddenSize, config.Model.NumHeads)
	log.Info().Msgf("Performance: %d CPU cores, %d MB memory limit", 
		config.Performance.CPUCores, config.Performance.MemoryLimitMB)
	log.Info().Msgf("Offline mode: %v", *offlineMode)
}

func setupComponents(ctx context.Context, config *Config) (*Components, error) {
	// ایجاد مدل
	modelInstance := model.NewNanoTransformer(config.Model)
	
	// ایجاد سیستم حافظه
	memorySystem, err := memory.NewDualMemory(config.Memory)
	if err != nil {
		return nil, fmt.Errorf("failed to create memory system: %w", err)
	}
	
	// ایجاد موتور جستجو
	searchEngine := search.NewMultiSearcher(config.Search)
	if *offlineMode {
		searchEngine.SetOfflineMode(true)
	}
	
	// ایجاد سیستم یادگیری
	learningSystem := learning.NewIncrementalLearner(
		modelInstance,
		memorySystem,
		config.Learning,
	)
	
	// بارگذاری دانش آفلاین
	if config.Offline.Enabled {
		if err := memorySystem.LoadOfflineKnowledge(config.Offline.KnowledgeBasePath); err != nil {
			log.Warn().Err(err).Msg("Failed to load offline knowledge")
		}
	}
	
	return &Components{
		Model:    modelInstance,
		Memory:   memorySystem,
		Search:   searchEngine,
		Learning: learningSystem,
	}, nil
}

func trainInitialModel(model *model.NanoTransformer, dataPath string) error {
	log.Info().Msg("Starting initial training with 10,000 samples")
	
	// بارگذاری داده‌های آموزشی
	dataset, err := model.LoadTrainingDataset(dataPath)
	if err != nil {
		return fmt.Errorf("failed to load training data: %w", err)
	}
	
	// آموزش مدل
	callbacks := []model.TrainingCallback{
		&model.ProgressCallback{},
		&model.CheckpointCallback{Interval: 1000},
		&model.EarlyStoppingCallback{Patience: 5},
	}
	
	model.TrainOnDataset(dataset, 3, callbacks...)
	
	// ذخیره مدل آموزش‌دیده
	if err := model.SaveCheckpoint("data/models/pretrained_10k.bin"); err != nil {
		return fmt.Errorf("failed to save trained model: %w", err)
	}
	
	log.Info().Msg("Initial training completed successfully")
	return nil
}

func startServices(ctx context.Context, config *Config, components *Components) (*Services, error) {
	services := &Services{}
	
	// سرویس سلامت
	healthService := NewHealthService(components)
	go healthService.Run(ctx)
	services.Health = healthService
	
	// سرویس آرشیو
	if config.Memory.CompressionLevel > 0 {
		archiveService := NewArchiveService(components.Memory, config.Memory)
		go archiveService.Run(ctx)
		services.Archive = archiveService
	}
	
	// سرویس پاک‌سازی حافظه
	cleanupService := NewCleanupService(components.Memory, config.Memory.RetentionDays)
	go cleanupService.Run(ctx)
	services.Cleanup = cleanupService
	
	return services, nil
}

func startIncrementalLearning(ctx context.Context, components *Components) {
	ticker := time.NewTicker(30 * time.Minute)
	defer ticker.Stop()
	
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			// بررسی آیا داده جدیدی برای یادگیری وجود دارد
			if components.Memory.HasNewSamples(100) {
				log.Info().Msg("Starting incremental learning cycle")
				
				samples := components.Memory.GetRecentSamples(1000)
				if err := components.Learning.LearnBatch(samples); err != nil {
					log.Error().Err(err).Msg("Incremental learning failed")
				} else {
					log.Info().Msg("Incremental learning completed")
				}
			}
		}
	}
}

func collectMetrics(ctx context.Context, components *Components) {
	ticker := time.NewTicker(60 * time.Second)
	defer ticker.Stop()
	
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			// جمع‌آوری آمار
			stats := components.Memory.GetStats()
			modelStats := components.Model.GetStats()
			searchStats := components.Search.GetStats()
			
			// نمایش آمار
			log.Debug().
				Int("memory_usage_mb", stats.MemoryUsageMB).
				Int("conversations", stats.TotalConversations).
				Int("knowledge_nodes", stats.KnowledgeNodes).
				Int("model_params_millions", modelStats.ParamsMillions).
				Float64("model_loss", modelStats.CurrentLoss).
				Int("search_queries", searchStats.TotalQueries).
				Int("cache_hits", searchStats.CacheHits).
				Msg("System metrics")
		}
	}
}

func shutdown(apiServer *api.Server, services *Services, components *Components) {
	log.Info().Msg("🛑 Starting graceful shutdown...")
	
	// توقف API سرور
	if apiServer != nil {
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		
		if err := apiServer.Shutdown(ctx); err != nil {
			log.Error().Err(err).Msg("Failed to shutdown API server gracefully")
		}
	}
	
	// ذخیره حالت فعلی
	log.Info().Msg("Saving current state...")
	
	// ذخیره مدل
	if err := components.Model.SaveCheckpoint("data/models/latest.bin"); err != nil {
		log.Error().Err(err).Msg("Failed to save model checkpoint")
	}
	
	// ذخیره حافظه
	if err := components.Memory.Flush(); err != nil {
		log.Error().Err(err).Msg("Failed to flush memory to disk")
	}
	
	// بستن اتصالات
	components.Search.Close()
	components.Memory.Close()
	
	log.Info().Msg("Shutdown sequence completed")
}

// تعاریف انواع
type Components struct {
	Model    *model.NanoTransformer
	Memory   *memory.DualMemory
	Search   *search.MultiSearcher
	Learning *learning.IncrementalLearner
}

type Services struct {
	Health   *HealthService
	Archive  *ArchiveService
	Cleanup  *CleanupService
}