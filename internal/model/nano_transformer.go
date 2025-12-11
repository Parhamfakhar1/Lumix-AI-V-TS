// internal/model/nano_transformer.go
package model

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"
	
	"github.com/Parhamfakhar1/Lumix-AI-V-TS/vts/internal/core"
	"github.com/rs/zerolog/log"
)

// NanoTransformer - مدل اصلی با پارامترهای کمینه‌شده
type NanoTransformer struct {
	config        Config
	embedding     *core.Tensor
	positionEnc   *core.Tensor
	layers        []*TransformerLayer
	outputLayer   *core.Tensor
	norm          *LayerNorm
	vocab         *Vocabulary
	tokenizer     *BPETokenizer
	optimizer     *core.AdamOptimizer
	scheduler     *core.CosineScheduler
	isTraining    bool
	trainingStats TrainingStats
	mu            sync.RWMutex
}

type Config struct {
	VocabSize      int     `json:"vocab_size"`
	HiddenSize     int     `json:"hidden_size"`
	NumLayers      int     `json:"num_layers"`
	NumHeads       int     `json:"num_heads"`
	MaxSeqLength   int     `json:"max_seq_length"`
	Dropout        float32 `json:"dropout"`
	LearningRate   float32 `json:"learning_rate"`
	BatchSize      int     `json:"batch_size"`
	WarmupSteps    int     `json:"warmup_steps"`
	WeightDecay    float32 `json:"weight_decay"`
	Quantization   bool    `json:"quantization"`
	Pruning        bool    `json:"pruning"`
}

type TransformerLayer struct {
	attention *core.LightMultiHeadAttention
	ffn       *FeedForwardNetwork
	norm1     *LayerNorm
	norm2     *LayerNorm
	dropout   float32
}

type FeedForwardNetwork struct {
	linear1 *core.Tensor
	linear2 *core.Tensor
	activation func(*core.Tensor) *core.Tensor
}

type LayerNorm struct {
	gamma *core.Tensor
	beta  *core.Tensor
	eps   float32
}

func NewNanoTransformer(config Config) *NanoTransformer {
	// مقداردهی اولیه توکن‌های ویژه
	vocab := NewVocabulary(config.VocabSize)
	vocab.AddSpecialTokens([]string{
		"[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
		"[BOS]", "[EOS]", "[USER]", "[ASSISTANT]",
	})
	
	// ایجاد مدل
	model := &NanoTransformer{
		config:      config,
		vocab:       vocab,
		tokenizer:   NewBPETokenizer(vocab),
		isTraining:  false,
	}
	
	// مقداردهی وزن‌ها
	model.initializeWeights()
	
	// ایجاد بهینه‌ساز
	model.optimizer = core.NewAdamOptimizer(
		config.LearningRate,
		0.9,  // beta1
		0.999, // beta2
		1e-8,  // epsilon
		config.WeightDecay,
	)
	
	// ایجاد زمان‌بند نرخ یادگیری
	model.scheduler = core.NewCosineScheduler(
		config.LearningRate,
		config.WarmupSteps,
		0.1, // min_lr_ratio
	)
	
	return model
}

func (nt *NanoTransformer) initializeWeights() {
	// Embedding layer
	nt.embedding = core.NewTensor([]int{nt.config.VocabSize, nt.config.HiddenSize}, core.DeviceCPU)
	core.XavierUniform(nt.embedding, float32(nt.config.HiddenSize))
	
	// Positional encoding
	nt.positionEnc = nt.createPositionalEncoding()
	
	// Transformer layers
	nt.layers = make([]*TransformerLayer, nt.config.NumLayers)
	for i := range nt.layers {
		nt.layers[i] = &TransformerLayer{
			attention: core.NewLightMultiHeadAttention(
				nt.config.HiddenSize,
				nt.config.NumHeads,
				nt.config.Dropout,
			),
			ffn: &FeedForwardNetwork{
				linear1: core.NewTensor([]int{nt.config.HiddenSize, nt.config.HiddenSize * 4}, core.DeviceCPU),
				linear2: core.NewTensor([]int{nt.config.HiddenSize * 4, nt.config.HiddenSize}, core.DeviceCPU),
				activation: core.GELU,
			},
			norm1: &LayerNorm{
				gamma: core.Ones([]int{nt.config.HiddenSize}),
				beta:  core.Zeros([]int{nt.config.HiddenSize}),
				eps:   1e-5,
			},
			norm2: &LayerNorm{
				gamma: core.Ones([]int{nt.config.HiddenSize}),
				beta:  core.Zeros([]int{nt.config.HiddenSize}),
				eps:   1e-5,
			},
			dropout: nt.config.Dropout,
		}
		
		// مقداردهی وزن‌های FFN
		core.KaimingUniform(nt.layers[i].ffn.linear1, "relu")
		core.XavierUniform(nt.layers[i].ffn.linear2, float32(nt.config.HiddenSize))
	}
	
	// Output layer
	nt.outputLayer = core.NewTensor([]int{nt.config.HiddenSize, nt.config.VocabSize}, core.DeviceCPU)
	core.XavierUniform(nt.outputLayer, float32(nt.config.HiddenSize))
	
	// Final layer norm
	nt.norm = &LayerNorm{
		gamma: core.Ones([]int{nt.config.HiddenSize}),
		beta:  core.Zeros([]int{nt.config.HiddenSize}),
		eps:   1e-5,
	}
}

func (nt *NanoTransformer) createPositionalEncoding() *core.Tensor {
	pe := core.NewTensor([]int{nt.config.MaxSeqLength, nt.config.HiddenSize}, core.DeviceCPU)
	
	for pos := 0; pos < nt.config.MaxSeqLength; pos++ {
		for i := 0; i < nt.config.HiddenSize; i++ {
			if i%2 == 0 {
				// سینوسی
				pe.Data[pos*nt.config.HiddenSize+i] = float32(math.Sin(
					float64(pos) / math.Pow(10000, float64(i)/float64(nt.config.HiddenSize)),
				))
			} else {
				// کسینوسی
				pe.Data[pos*nt.config.HiddenSize+i] = float32(math.Cos(
					float64(pos) / math.Pow(10000, float64(i-1)/float64(nt.config.HiddenSize)),
				))
			}
		}
	}
	
	return pe
}

func (nt *NanoTransformer) Forward(inputIDs []int, attentionMask *core.Tensor) (*core.Tensor, *core.Tensor) {
	nt.mu.RLock()
	defer nt.mu.RUnlock()
	
	batchSize := 1
	seqLen := len(inputIDs)
	
	if seqLen > nt.config.MaxSeqLength {
		seqLen = nt.config.MaxSeqLength
		inputIDs = inputIDs[:seqLen]
	}
	
	// Token embeddings
	tokenEmbeddings := nt.getEmbeddings(inputIDs)
	
	// Position embeddings
	positionIDs := make([]int, seqLen)
	for i := range positionIDs {
		positionIDs[i] = i
	}
	posEmbeddings := nt.getPositionEmbeddings(positionIDs)
	
	// Combine embeddings
	embeddings := tokenEmbeddings.Add(posEmbeddings)
	
	// Apply dropout if training
	if nt.isTraining && nt.config.Dropout > 0 {
		embeddings = embeddings.Dropout(nt.config.Dropout)
	}
	
	// Transformer layers
	hiddenStates := embeddings
	for _, layer := range nt.layers {
		// Self-attention
		attnOutput := layer.attention.Forward(
			hiddenStates, hiddenStates, hiddenStates,
			attentionMask, "",
		)
		
		// Add & Norm
		hiddenStates = layer.norm1.Forward(
			hiddenStates.Add(attnOutput),
		)
		
		// Feed-forward
		ffnOutput := layer.ffn.linear1.MatMul(hiddenStates)
		ffnOutput = layer.ffn.activation(ffnOutput)
		ffnOutput = layer.ffn.linear2.MatMul(ffnOutput)
		
		// Add & Norm
		hiddenStates = layer.norm2.Forward(
			hiddenStates.Add(ffnOutput),
		)
		
		// Apply dropout
		if nt.isTraining && layer.dropout > 0 {
			hiddenStates = hiddenStates.Dropout(layer.dropout)
		}
	}
	
	// Final normalization
	hiddenStates = nt.norm.Forward(hiddenStates)
	
	// Output projection
	logits := hiddenStates.MatMul(nt.outputLayer)
	
	return logits, hiddenStates
}

func (nt *NanoTransformer) TrainOnDataset(dataset *TrainingDataset, epochs int, callbacks ...TrainingCallback) {
	nt.mu.Lock()
	nt.isTraining = true
	nt.mu.Unlock()
	
	defer func() {
		nt.mu.Lock()
		nt.isTraining = false
		nt.mu.Unlock()
	}()
	
	log.Info().Msgf("Starting training on %d samples", dataset.Size())
	
	totalSteps := epochs * (dataset.Size() / nt.config.BatchSize)
	step := 0
	
	for epoch := 0; epoch < epochs; epoch++ {
		log.Info().Msgf("Epoch %d/%d", epoch+1, epochs)
		
		// Shuffle dataset
		dataset.Shuffle()
		
		// Create batches
		batches := dataset.Batch(nt.config.BatchSize)
		
		for batchIdx, batch := range batches {
			step++
			
			// Forward pass
			logits, _ := nt.Forward(batch.InputIDs, batch.AttentionMask)
			
			// Calculate loss
			loss := nt.calculateLoss(logits, batch.TargetIDs)
			
			// Backward pass
			nt.backward(loss)
			
			// Optimizer step
			nt.optimizer.Step(nt.parameters())
			
			// Update learning rate
			lr := nt.scheduler.GetLR(step)
			nt.optimizer.SetLR(lr)
			
			// Update statistics
			nt.trainingStats.Update(loss.Value(), step, lr)
			
			// Callbacks
			for _, cb := range callbacks {
				cb.OnBatchEnd(batchIdx, loss.Value(), nt.trainingStats)
			}
			
			// Log progress
			if step%100 == 0 {
				log.Info().Msgf(
					"Step %d/%d - Loss: %.4f - LR: %.6f",
					step, totalSteps, loss.Value(), lr,
				)
			}
			
			// Save checkpoint
			if step%nt.config.CheckpointInterval == 0 {
				nt.SaveCheckpoint(fmt.Sprintf("checkpoint_step_%d.bin", step))
			}
		}
		
		// Validation
		if dataset.HasValidation() {
			valLoss := nt.validate(dataset.ValidationSet())
			log.Info().Msgf("Validation Loss: %.4f", valLoss)
			
			for _, cb := range callbacks {
				cb.OnEpochEnd(epoch, valLoss, nt.trainingStats)
			}
		}
	}
	
	log.Info().Msg("Training completed")
}

func (nt *NanoTransformer) Generate(prompt string, maxLength int, temperature float32, 
	topK int, topP float32, useSearch bool, searchResults []SearchResult) string {
	
	nt.mu.RLock()
	defer nt.mu.RUnlock()
	
	// Tokenize prompt
	tokens := nt.tokenizer.Encode(prompt)
	
	// Add search context if available
	if useSearch && len(searchResults) > 0 {
		context := nt.prepareSearchContext(searchResults)
		tokens = append(nt.tokenizer.Encode(context), tokens...)
		
		// Truncate if too long
		if len(tokens) > nt.config.MaxSeqLength/2 {
			tokens = tokens[:nt.config.MaxSeqLength/2]
		}
	}
	
	// Add special tokens
	tokens = append([]int{nt.vocab.TokenToID("[BOS]")}, tokens...)
	
	// Generate tokens
	for len(tokens) < maxLength && len(tokens) < nt.config.MaxSeqLength {
		// Get model predictions
		logits, _ := nt.Forward(tokens, nil)
		
		// Get last token logits
		lastLogits := logits.Slice([]int{0, len(tokens)-1, 0}, []int{1, len(tokens), nt.config.VocabSize})
		
		// Apply temperature
		if temperature != 1.0 {
			lastLogits = lastLogits.Div(core.Scalar(temperature))
		}
		
		// Apply top-k/top-p sampling
		probs := lastLogits.Softmax(-1)
		if topK > 0 {
			probs = probs.TopK(topK)
		}
		if topP > 0 {
			probs = probs.TopP(topP)
		}
		
		// Sample next token
		nextToken := core.SampleCategorical(probs)
		
		// Check for EOS token
		if nextToken == nt.vocab.TokenToID("[EOS]") {
			break
		}
		
		// Add token to sequence
		tokens = append(tokens, nextToken)
	}
	
	// Decode tokens to text
	generated := nt.tokenizer.Decode(tokens)
	
	return generated
}

func (nt *NanoTransformer) SaveCheckpoint(path string) error {
	nt.mu.Lock()
	defer nt.mu.Unlock()
	
	// Create directory if not exists
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return err
	}
	
	// Prepare checkpoint data
	checkpoint := Checkpoint{
		Config:        nt.config,
		Version:       "1.0.0",
		Step:          nt.trainingStats.Step,
		TrainingStats: nt.trainingStats,
		Timestamp:     time.Now().Unix(),
	}
	
	// Save metadata
	metaPath := path + ".meta"
	metaFile, err := os.Create(metaPath)
	if err != nil {
		return err
	}
	defer metaFile.Close()
	
	encoder := json.NewEncoder(metaFile)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(checkpoint); err != nil {
		return err
	}
	
	// Save model weights
	weightsFile, err := os.Create(path)
	if err != nil {
		return err
	}
	defer weightsFile.Close()
	
	// Save all parameters
	params := nt.parameters()
	
	// Apply quantization if enabled
	if nt.config.Quantization {
		params = nt.quantizeParameters(params)
	}
	
	// Save parameters
	if err := core.SaveTensors(weightsFile, params); err != nil {
		return err
	}
	
	log.Info().Msgf("Checkpoint saved: %s", path)
	return nil
}

func (nt *NanoTransformer) LoadCheckpoint(path string) error {
	nt.mu.Lock()
	defer nt.mu.Unlock()
	
	// Load metadata
	metaPath := path + ".meta"
	metaFile, err := os.Open(metaPath)
	if err != nil {
		return err
	}
	defer metaFile.Close()
	
	var checkpoint Checkpoint
	decoder := json.NewDecoder(metaFile)
	if err := decoder.Decode(&checkpoint); err != nil {
		return err
	}
	
	// Verify config compatibility
	if !nt.config.Compatible(checkpoint.Config) {
		return fmt.Errorf("incompatible model configuration")
	}
	
	// Load weights
	weightsFile, err := os.Open(path)
	if err != nil {
		return err
	}
	defer weightsFile.Close()
	
	params, err := core.LoadTensors(weightsFile)
	if err != nil {
		return err
	}
	
	// Apply dequantization if needed
	if checkpoint.Config.Quantization {
		params = nt.dequantizeParameters(params)
	}
	
	// Load parameters into model
	nt.loadParameters(params)
	
	// Update training stats
	nt.trainingStats = checkpoint.TrainingStats
	
	log.Info().Msgf("Checkpoint loaded: %s (step: %d)", path, checkpoint.Step)
	return nil
}

func (nt *NanoTransformer) prepareSearchContext(results []SearchResult) string {
	var context strings.Builder
	context.WriteString("جستجوی اینترنتی انجام شد. اطلاعات یافت شده:\n\n")
	
	for i, result := range results {
		context.WriteString(fmt.Sprintf("%d. %s\n", i+1, result.Title))
		context.WriteString(fmt.Sprintf("   %s\n", result.Snippet))
		
		if result.Summary != "" {
			context.WriteString(fmt.Sprintf("   خلاصه: %s\n", result.Summary))
		}
		
		context.WriteString("\n")
	}
	
	return context.String()
}