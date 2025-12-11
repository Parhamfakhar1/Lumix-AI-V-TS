// internal/memory/advanced_memory.go
package memory

import (
	"encoding/binary"
	"fmt"
	"hash/fnv"
	"sort"
	"sync"
	"time"
	
	"github.com/lumix-ai/vts/internal/core"
	"github.com/klauspost/compress/zstd"
)

// NeuralMemory - حافظه عصبی برای یادگیری عمیق‌تر
type NeuralMemory struct {
	AssociativeGraph *AssociativeGraph
	EpisodicMemory   *EpisodicStore
	SemanticMemory   *SemanticNetwork
	ProceduralMemory *ProceduralStore
	WorkingMemory    *WorkingBuffer
	Consolidator     *MemoryConsolidator
}

// AssociativeGraph - گراف تداعی‌های مفهومی
type AssociativeGraph struct {
	nodes map[string]*ConceptNode
	edges map[string]*AssociationEdge
	mu    sync.RWMutex
}

type ConceptNode struct {
	ID           string
	Label        string
	Embedding    []float32
	Strength     float32
	LastAccessed time.Time
	AccessCount  int
	RelatedConcepts map[string]float32 // conceptID -> strength
	Properties   map[string]interface{}
}

type AssociationEdge struct {
	From     string
	To       string
	Type     string  // "is-a", "has", "related", "causes"
	Strength float32
	Weight   float32
	Evidence int     // تعداد دفعات مشاهده
}

func NewNeuralMemory() *NeuralMemory {
	return &NeuralMemory{
		AssociativeGraph: &AssociativeGraph{
			nodes: make(map[string]*ConceptNode),
			edges: make(map[string]*AssociationEdge),
		},
		EpisodicMemory:   NewEpisodicStore(),
		SemanticMemory:   NewSemanticNetwork(),
		ProceduralMemory: NewProceduralStore(),
		WorkingMemory:    NewWorkingBuffer(100), // 100 آیتم در حافظه کاری
		Consolidator:     NewMemoryConsolidator(),
	}
}

// یادگیری تداعی جدید
func (nm *NeuralMemory) LearnAssociation(conceptA, conceptB, relationType string, strength float32) {
	nm.mu.Lock()
	defer nm.mu.Unlock()
	
	// ایجاد یا به‌روزرسانی گره‌ها
	nodeA := nm.getOrCreateNode(conceptA)
	nodeB := nm.getOrCreateNode(conceptB)
	
	// ایجاد یا تقویت یال
	edgeID := nm.generateEdgeID(conceptA, conceptB, relationType)
	if edge, exists := nm.edges[edgeID]; exists {
		// تقویت اتصال موجود
		edge.Strength = (edge.Strength + strength) / 2
		edge.Evidence++
		edge.Weight = edge.Strength * float32(edge.Evidence)
	} else {
		// ایجاد اتصال جدید
		nm.edges[edgeID] = &AssociationEdge{
			From:     conceptA,
			To:       conceptB,
			Type:     relationType,
			Strength: strength,
			Weight:   strength,
			Evidence: 1,
		}
	}
	
	// به‌روزرساری گره‌ها
	nodeA.RelatedConcepts[conceptB] = strength
	nodeB.RelatedConcepts[conceptA] = strength
	
	// تثبیت حافظه
	nm.consolidateIfNeeded()
}

// استنتاج بر اساس تداعی‌ها
func (nm *NeuralMemory) Infer(concept string, depth int) []InferenceResult {
	nm.mu.RLock()
	defer nm.mu.RUnlock()
	
	node, exists := nm.nodes[concept]
	if !exists {
		return nil
	}
	
	var results []InferenceResult
	visited := make(map[string]bool)
	
	nm.traverseAssociations(node, depth, 1.0, visited, &results)
	
	// مرتب‌سازی بر اساس قدرت استنتاج
	sort.Slice(results, func(i, j int) bool {
		return results[i].Confidence > results[j].Confidence
	})
	
	return results
}

// پیمایش بازگشتی گراف تداعی
func (nm *NeuralMemory) traverseAssociations(node *ConceptNode, depth int, 
	pathStrength float32, visited map[string]bool, results *[]InferenceResult) {
	
	if depth <= 0 || visited[node.ID] {
		return
	}
	
	visited[node.ID] = true
	
	// بررسی تمام یال‌های خروجی
	for _, edge := range nm.getEdgesFrom(node.ID) {
		nextNode, exists := nm.nodes[edge.To]
		if !exists {
			continue
		}
		
		// محاسبه اطمینان استنتاج
		inferenceStrength := pathStrength * edge.Strength
		
		// اضافه کردن نتیجه
		*results = append(*results, InferenceResult{
			Concept:    nextNode.Label,
			Relation:   edge.Type,
			Confidence: inferenceStrength,
			PathLength: 5 - depth, // عمق معکوس
		})
		
		// ادامه پیمایش
		nm.traverseAssociations(nextNode, depth-1, inferenceStrength, visited, results)
	}
}