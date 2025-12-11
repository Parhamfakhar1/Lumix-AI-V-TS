// internal/core/tensor.go
package core

import (
	"encoding/binary"
	"fmt"
	"math"
	"sync"
)

// Tensor - ساختار بهینه‌شده برای CPU ضعیف
type Tensor struct {
	Data  []float32
	Shape []int
	Stride []int
	Offset int
	requiresGrad bool
	grad *Tensor
	device Device
}

type Device string

const (
	DeviceCPU Device = "cpu"
	DeviceAuto Device = "auto"
)

// NewTensor - ایجاد تانسور جدید با مدیریت حافظه هوشمند
func NewTensor(shape []int, device Device) *Tensor {
	size := 1
	stride := make([]int, len(shape))
	currentStride := 1
	
	for i := len(shape) - 1; i >= 0; i-- {
		stride[i] = currentStride
		size *= shape[i]
		currentStride *= shape[i]
	}
	
	// Align memory for better cache performance
	alignedSize := ((size + 7) / 8) * 8
	
	return &Tensor{
		Data:  make([]float32, alignedSize),
		Shape: shape,
		Stride: stride,
		device: device,
	}
}

// MatMul - ضرب ماتریس بهینه‌شده با حافظه پنهان
func (t *Tensor) MatMul(other *Tensor) (*Tensor, error) {
	if len(t.Shape) != 2 || len(other.Shape) != 2 {
		return nil, fmt.Errorf("matmul requires 2D tensors")
	}
	
	if t.Shape[1] != other.Shape[0] {
		return nil, fmt.Errorf("shape mismatch: %v @ %v", t.Shape, other.Shape)
	}
	
	m, n, p := t.Shape[0], t.Shape[1], other.Shape[1]
	result := NewTensor([]int{m, p}, t.device)
	
	// بلوک‌بندی برای بهینه‌سازی حافظه پنهان
	blockSize := 8 // مناسب برای CPU ضعیف
	var wg sync.WaitGroup
	
	for i := 0; i < m; i += blockSize {
		for j := 0; j < p; j += blockSize {
			wg.Add(1)
			go func(iStart, jStart int) {
				defer wg.Done()
				
				iEnd := min(iStart+blockSize, m)
				jEnd := min(jStart+blockSize, p)
				
				for ii := iStart; ii < iEnd; ii++ {
					for jj := jStart; jj < jEnd; jj++ {
						sum := float32(0)
						// Loop unrolling برای سرعت بیشتر
						kk := 0
						for ; kk+3 < n; kk += 4 {
							sum += t.Data[ii*t.Stride[0]+kk] * other.Data[kk*other.Stride[0]+jj] +
								t.Data[ii*t.Stride[0]+kk+1] * other.Data[(kk+1)*other.Stride[0]+jj] +
								t.Data[ii*t.Stride[0]+kk+2] * other.Data[(kk+2)*other.Stride[0]+jj] +
								t.Data[ii*t.Stride[0]+kk+3] * other.Data[(kk+3)*other.Stride[0]+jj]
						}
						for ; kk < n; kk++ {
							sum += t.Data[ii*t.Stride[0]+kk] * other.Data[kk*other.Stride[0]+jj]
						}
						result.Data[ii*result.Stride[0]+jj] = sum
					}
				}
			}(i, j)
		}
	}
	
	wg.Wait()
	return result, nil
}

// QuantizeINT8 - تبدیل به 8-bit برای صرفه‌جویی در حافظه
func (t *Tensor) QuantizeINT8() ([]int8, float32, float32) {
	if len(t.Data) == 0 {
		return []int8{}, 0, 0
	}
	
	// پیدا کردن min/max برای مقیاس‌گذاری
	minVal := t.Data[0]
	maxVal := t.Data[0]
	for _, v := range t.Data {
		if v < minVal {
			minVal = v
		}
		if v > maxVal {
			maxVal = v
		}
	}
	
	scale := (maxVal - minVal) / 255.0
	zeroPoint := -minVal / scale
	
	quantized := make([]int8, len(t.Data))
	for i, v := range t.Data {
		q := int8(math.Round(float64((v - minVal) / scale)))
		quantized[i] = q
	}
	
	return quantized, scale, zeroPoint
}

// DequantizeINT8 - بازسازی از 8-bit
func DequantizeINT8(quantized []int8, scale, zeroPoint float32) *Tensor {
	t := NewTensor([]int{len(quantized)}, DeviceCPU)
	for i, q := range quantized {
		t.Data[i] = float32(q)*scale + zeroPoint
	}
	return t
}

// ApplyPruning - هرس کردن وزن‌های کم‌اهمیت
func (t *Tensor) ApplyPruning(sparsity float32) *Tensor {
	if sparsity <= 0 || sparsity >= 1 {
		return t
	}
	
	// محاسبه آستانه بر اساس مطلق مقادیر
	absValues := make([]float32, len(t.Data))
	for i, v := range t.Data {
		absValues[i] = float32(math.Abs(float64(v)))
	}
	
	// یافتن صدک مورد نظر
	threshold := percentile(absValues, sparsity)
	
	// صفر کردن وزن‌های زیر آستانه
	pruned := NewTensor(t.Shape, t.device)
	copy(pruned.Data, t.Data)
	
	zeroed := 0
	for i, v := range t.Data {
		if float32(math.Abs(float64(v))) < threshold {
			pruned.Data[i] = 0
			zeroed++
		}
	}
	
	return pruned
}

// SaveBinary - ذخیره بهینه در فایل باینری
func (t *Tensor) SaveBinary(filename string) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()
	
	// هدر فایل
	header := make([]byte, 16)
	binary.LittleEndian.PutUint32(header[0:4], 0x4C554D58) // "LUMX"
	binary.LittleEndian.PutUint32(header[4:8], 1)          // Version
	binary.LittleEndian.PutUint32(header[8:12], uint32(len(t.Shape)))
	
	// نوشتن shape
	for _, dim := range t.Shape {
		binary.Write(file, binary.LittleEndian, int32(dim))
	}
	
	// نوشتن داده‌ها با فشرده‌سازی
	compressed, err := compressFloat32(t.Data)
	if err != nil {
		return err
	}
	
	binary.Write(file, binary.LittleEndian, int32(len(compressed)))
	file.Write(compressed)
	
	return nil
}

// توابع کمکی
func percentile(values []float32, p float32) float32 {
	sorted := make([]float32, len(values))
	copy(sorted, values)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })
	
	idx := int(p * float32(len(sorted)-1))
	if idx >= len(sorted) {
		idx = len(sorted) - 1
	}
	return sorted[idx]
}

func compressFloat32(data []float32) ([]byte, error) {
	// فشرده‌سازی ساده برای CPU ضعیف
	buf := new(bytes.Buffer)
	for _, v := range data {
		// تبدیل به int16 با مقیاس‌گذاری
		scaled := int16(v * 32767.0)
		binary.Write(buf, binary.LittleEndian, scaled)
	}
	return buf.Bytes(), nil
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}