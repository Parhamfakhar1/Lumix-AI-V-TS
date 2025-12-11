// internal/core/attention.go
package core

import (
	"math"
)

// LightMultiHeadAttention - توجه چندسر بهینه‌شده
type LightMultiHeadAttention struct {
	numHeads   int
	headDim    int
	scale      float32
	dropout    float32
	Wq, Wk, Wv *Tensor
	Wo         *Tensor
	cacheEnabled bool
	kCache, vCache map[string]*Tensor
}

func NewLightMultiHeadAttention(hiddenSize, numHeads int, dropout float32) *LightMultiHeadAttention {
	headDim := hiddenSize / numHeads
	
	return &LightMultiHeadAttention{
		numHeads:   numHeads,
		headDim:    headDim,
		scale:      1.0 / float32(math.Sqrt(float64(headDim))),
		dropout:    dropout,
		Wq:        NewTensor([]int{hiddenSize, hiddenSize}, DeviceCPU),
		Wk:        NewTensor([]int{hiddenSize, hiddenSize}, DeviceCPU),
		Wv:        NewTensor([]int{hiddenSize, hiddenSize}, DeviceCPU),
		Wo:        NewTensor([]int{hiddenSize, hiddenSize}, DeviceCPU),
		cacheEnabled: true,
		kCache:     make(map[string]*Tensor),
		vCache:     make(map[string]*Tensor),
	}
}

func (mha *LightMultiHeadAttention) Forward(query, key, value *Tensor, mask *Tensor, cacheKey string) *Tensor {
	batchSize := query.Shape[0]
	seqLen := query.Shape[1]
	
	// خطی‌سازی برای توجه چندسر
	q := query.MatMul(mha.Wq) // [batch, seq_len, hidden]
	k := key.MatMul(mha.Wk)   // [batch, seq_len, hidden]
	v := value.MatMul(mha.Wv) // [batch, seq_len, hidden]
	
	// تغییر شکل برای توجه چندسر
	q = mha.splitHeads(q, batchSize, seqLen)
	k = mha.splitHeads(k, batchSize, seqLen)
	v = mha.splitHeads(v, batchSize, seqLen)
	
	// استفاده از کش اگر فعال باشد
	if mha.cacheEnabled && cacheKey != "" {
		if cachedK, ok := mha.kCache[cacheKey]; ok {
			// الحاق با کش قدیمی
			k = mha.concatCache(cachedK, k)
			v = mha.concatCache(mha.vCache[cacheKey], v)
		}
		// به‌روزرسانی کش
		mha.kCache[cacheKey] = k
		mha.vCache[cacheKey] = v
	}
	
	// محاسبه توجه
	scores := mha.attention(q, k, v, mask)
	
	// ترکیب سرها
	output := mha.combineHeads(scores, batchSize, seqLen)
	
	// لایه خروجی
	output = output.MatMul(mha.Wo)
	
	return output
}

func (mha *LightMultiHeadAttention) attention(q, k, v, mask *Tensor) *Tensor {
	// Q * K^T
	scores, _ := q.MatMul(k.Transpose())
	
	// Scale
	scores = scores.Scale(mha.scale)
	
	// اعمال ماسک (اگر وجود دارد)
	if mask != nil {
		scores = scores.Add(mask.Neg())
	}
	
	// Softmax
	probs := scores.Softmax(-1)
	
	// Dropout (فقط در آموزش)
	if mha.dropout > 0 && mha.training {
		probs = probs.Dropout(mha.dropout)
	}
	
	// توجه * مقادیر
	output, _ := probs.MatMul(v)
	
	return output
}

func (mha *LightMultiHeadAttention) splitHeads(x *Tensor, batchSize, seqLen int) *Tensor {
	// تغییر شکل: [batch, seq_len, hidden] -> [batch, num_heads, seq_len, head_dim]
	newShape := []int{batchSize, seqLen, mha.numHeads, mha.headDim}
	reshaped := x.Reshape(newShape)
	
	// جابجایی محورها: [batch, seq_len, num_heads, head_dim] -> [batch, num_heads, seq_len, head_dim]
	return reshaped.Transpose(1, 2)
}

func (mha *LightMultiHeadAttention) combineHeads(x *Tensor, batchSize, seqLen int) *Tensor {
	// جابجایی معکوس: [batch, num_heads, seq_len, head_dim] -> [batch, seq_len, num_heads, head_dim]
	x = x.Transpose(1, 2)
	
	// تغییر شکل به حالت اولیه: [batch, seq_len, hidden]
	newShape := []int{batchSize, seqLen, mha.numHeads * mha.headDim}
	return x.Reshape(newShape)
}

func (mha *LightMultiHeadAttention) concatCache(cached, new *Tensor) *Tensor {
	// الحاق در بعد seq_len
	batchSize := cached.Shape[0]
	numHeads := cached.Shape[1]
	cachedLen := cached.Shape[2]
	newLen := new.Shape[2]
	headDim := cached.Shape[3]
	
	combined := NewTensor([]int{batchSize, numHeads, cachedLen + newLen, headDim}, DeviceCPU)
	
	// کپی داده‌های کش‌شده
	copy(combined.Data[:cached.Size()], cached.Data)
	
	// اضافه کردن داده‌های جدید
	offset := cached.Size()
	copy(combined.Data[offset:offset+new.Size()], new.Data)
	
	return combined
}